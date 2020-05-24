import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import nn.BPNeuralNetwork;
import nn.DataAdapter;
import nn.InvalidArgumentVectorSize;
import nn.NoSuchLayerException;

public class DriverProgram {

	public static void main(final String[] args) {
		int totCount = 0, totCorr = 0;
		final double errAllowed = (double) 0.2;
		final int layer_strengths[] = { 50, 10, 2 };
		BPNeuralNetwork network = new BPNeuralNetwork(layer_strengths) {
			@Override
			public double transferFunc(double x) {
				double ex = Math.exp(-x);
				return (1 / (1 + ex));
			}

			@Override
			public double transferDeriv(double x) {
				return transferFunc(x) * (1 - transferFunc(x));
			}
		};
		network.setLearning_rate(1);
		network.setMomentum(0.002);

		File inpFile = new File("dlbcl-selected-50.csv"), oFile = new File("outFile");

		DataAdapter dAdapter;
		try {
			dAdapter = new DLBCLAdapter(new FileInputStream(inpFile), new FileOutputStream(oFile), 50, 2);
			network.setDataAdapter(dAdapter);
			network.configAllInitialValuesToRandom();
			dAdapter.setPartitionRatio((float)0.7);

			totCount = dAdapter.getInputs().size();

			network.trainNetwork();
			double testErrors[] = network.testNetwork();

			for (int i = 0; i < testErrors.length; i++) {
				totCorr += testErrors[i] < errAllowed ? 1 : 0;
				System.out.println("Test "+i+" RMS Error: "+testErrors[i]);
			}
			totCount = testErrors.length;
		} catch (FileNotFoundException e) {
			System.out.println("No such file is there.");
			e.printStackTrace();
		} catch (InvalidArgumentVectorSize e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (NoSuchLayerException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Total Correct Guess: " + totCorr + " out of " + totCount);
		System.out.println("Total Correct Guess Ratio: " + (double) totCorr / totCount);

	}

}