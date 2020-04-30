import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import nn.BPNeuralNetwork;
import nn.InvalidArgumentVectorSize;
import nn.NoSuchLayerException;
import nn.Perceptron;
import nn.PerceptronLayer;

public class DriverProgram {

	public static void main(String[] args) {
		int totCount=0, totCorr=0, inptSize;
		double errAllowed = (double) 0.1;
		// TODO Auto-generated method stub
		int layerCount = 5, layer_strengths[] = { 6, 8, 10, 5, 2 };
		BPNeuralNetwork network = new BPNeuralNetwork(layerCount, layer_strengths) {
			@Override
			public double transferFunc(double x) {
				double res= (double) (1 / (1 + Math.pow(Math.E, -x)));
				// return double.valueOf(res).isNaN() ? x : res;
				return res;
			}

			@Override
			public double transferDeriv(double x) {
				return (transferFunc(x) * (1 - transferFunc(x)));
			}
		};
		network.setLearning_rate(0.5);
		network.setMomentum(0.2);
		try {
			network.configAllInitialValuesToRandom();
		} catch (InvalidArgumentVectorSize e) {
			e.printStackTrace();
		}
		List<double[]> inputs = new ArrayList<>();
		List<double[]> outputs = new ArrayList<>();
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter Input filepath: ");
		try {
			FileInputStream fis = new FileInputStream(new File(sc.nextLine()));
			sc.close();
			sc = new Scanner(fis);
			while (sc.hasNext()) {
				double[] ip = new double[6];
				double[] op = new double[2];
				ip[0] = sc.nextDouble();
				ip[1] = sc.nextDouble();
				ip[2] = sc.nextDouble();
				ip[3] = sc.nextDouble();
				ip[4] = sc.nextDouble();
				ip[5] = sc.nextDouble();
				inputs.add(ip);
				op[0] = sc.nextDouble();
				op[1] = sc.nextDouble();
				outputs.add(op);
			}
			sc.close();

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (int i = 0; i < inputs.size(); i++) {
			double ips[] = inputs.get(i);
			inptSize = inputs.size();
			try {
				network.propagateInput(ips);
				double op[] = network.getNetworkOutput();
				System.out.println("Network Output: " + op[0] + " " + op[1] );
				if(i < inptSize * 0.7)
					network.calcAndBackPropagateError(outputs.get(i));
				else {
					if(errAllowed >( Math.abs(outputs.get(i)[0] - op[0]) +  Math.abs(outputs.get(i)[1] - op[1])))
						totCorr++;
					else
						System.out.println("WRONG GUESS");
					totCount++;
				}
				
					
			} catch (InvalidArgumentVectorSize e) {
				e.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (NoSuchLayerException e) {
				e.printStackTrace();
			}
			
		}
		System.out.println("Total Correct Guess: " + totCorr +" out of "+totCount);
		System.out.println("Total Correct Guess Ratio: " + (double) totCorr/totCount);
		

	}

}
