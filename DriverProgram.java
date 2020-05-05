import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import nn.BPNeuralNetwork;
import nn.DataAdapter;
import nn.InvalidArgumentVectorSize;
import nn.NoSuchLayerException;
import nn.Perceptron;
import nn.PerceptronLayer;

public class DriverProgram {

	public static void main(final String[] args) {
		int totCount = 0, totCorr = 0, inptSize;
		final double errAllowed = (double) 0.1;
		// TODO Auto-generated method stub
		final int layer_strengths[] = { 12, 24, 50,25, 5, 1 };
		BPNeuralNetwork network = new BPNeuralNetwork(layer_strengths) {
			@Override
			public double transferFunc(final double x) {
				final double res = (double) (1 / (1 + Math.pow(Math.E, -x)));
				// return double.valueOf(res).isNaN() ? x : res;
				return res;
			}

			@Override
			public double transferDeriv(final double x) {
				return (transferFunc(x) * (1 - transferFunc(x)));
			}
		};
		network.setLearning_rate(0.5);
		network.setMomentum(0);

		File inpFile = new File("Churn_Modelling.csv"), oFile = new File("outFile");

		MyAdapter dAdapter;
		try {
			dAdapter = new MyAdapter(new FileInputStream(inpFile), new FileOutputStream(oFile), 12, 1);
			// dAdapter.setRestrictedDataSize(5000);
			network.setDataAdapter(dAdapter);
			network.configAllInitialValuesToRandom();
			dAdapter.setPartitionRatio((float)0.7);

			totCount = dAdapter.getInputs().size();

			double trainingErrors[] = network.trainNetwork();
			for (int i = 0; i < trainingErrors.length; i++) {
				totCorr += trainingErrors[i] < errAllowed ? 1 : 0;
			}
			double testErrors[] = network.testNetwork();

			for (int i = 0; i < testErrors.length; i++) {
				totCorr += testErrors[i] < errAllowed ? 1 : 0;
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvalidArgumentVectorSize e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NoSuchLayerException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Total Correct Guess: " + totCorr + " out of " + totCount);
		System.out.println("Total Correct Guess Ratio: " + (double) totCorr / totCount);

	}

}

class MyAdapter extends DataAdapter {

	public MyAdapter(final InputStream inputStream, final OutputStream outputStream, final int inputColumnWidth, final int outputColumnWidth) {
		super(inputStream, outputStream, inputColumnWidth, outputColumnWidth);
	}

	@Override
	public String formattedOutputColumn(int columnPosition, double calcOutput) {
		if(columnPosition == 0){
			return String.valueOf(calcOutput);
		}
		return null;
	}

	double min_max_normalization(double value, double min, double max){
		return (value - min) / (max - min);
	}

	@Override
	public void normalizeInputRow(String row, double[] inputs, double[] outputs) {
		Scanner sc = new Scanner(row);
		sc.useDelimiter(",");
		System.out.print(sc.next()+ " ");
		System.out.print(sc.next()+ " ");
		System.out.print(sc.next()+ " ");
		inputs[0] = min_max_normalization(Double.valueOf(sc.next()), 350, 850);
		String country = sc.next();
		inputs[1] = country.equals("Germany") ? 1 : 0;
		inputs[2] = country.equals("France") ? 1 : 0;
		inputs[3] = country.equals("Spain") ? 1 : 0;
		String gender = sc.next();
		inputs[4] = gender.equals("Male") ? 1 : -1;
		inputs[5] = min_max_normalization(Double.valueOf(sc.next()), 18, 92);
		inputs[6] = min_max_normalization(Double.valueOf(sc.next()), 0, 10);
		inputs[7] = min_max_normalization(Double.valueOf(sc.next()), 0, 250898.09);
		inputs[8] = min_max_normalization(Double.valueOf(sc.next()), 1, 4);
		inputs[9] = Double.valueOf(sc.next());
		inputs[10] = Double.valueOf(sc.next());
		inputs[11] = min_max_normalization(Double.valueOf(sc.next()), 11.58, 199992.48);
		if(sc.hasNext())
		outputs[0] = Double.valueOf(sc.next());
		// System.out.println(sc.next());
		// while(sc.hasNext()){
		// 	System.out.println(sc.next());
		// }
		sc.close();
	}
	
}