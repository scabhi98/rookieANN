# rookieANN

Simple Impelementation of Backpropagation Algorithm on Java

## Getting Started

You need to extend ```BPNeuralNetwork``` class to your version by overriding two methods ```transferFunc()``` as a transfer function and derivative of transfer function as ```transferDeriv()``` .

### Example:

```java
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
```

Create a personalized DataAdapter by extending the ```DataAdapter``` class which can handle your dataset files.
You need to define two abstract functions, ```normalizeInputRow()``` to get a normalized numeric input row from input values, and  ```formattedOutputColumn()``` to format the numeric output in desired format.

Example

```java
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
```

Instantiate extended DataAdapter class and pass the following parameters
	An InputStream from where it will read its inputs, 
	An OutputStream where network will print outputs,
	An Integer value indicating network's input node count,
	An Integer value indicating network's output node count.

then set a partition ratio for the dataset. Default is 0.65 i.e. 65%.

```java
	MyAdapter dAdapter = new MyAdapter(new FileInputStream(inpFile), new FileOutputStream(oFile), 12, 1);
	dAdapter.setPartitionRatio((float)0.7);
```

Then Connect the network to the DataAdapter instantiated and initialize all network weights and bias to random as,

```java
	network.setDataAdapter(dAdapter);
	network.configAllInitialValuesToRandom();
```

All set. Now calling ```network.trainNetwork()``` will initiate training of the network feeding from DataAdapter which is connected to the input file.

```network.testNetwork()``` will run the Network against the input case without further backpropagating the errors i.e. it will give output only.

These two methods returns an array of double values which are collection of errors occured at output with the labled output in input file.

```java
	double trainingErrors[] = network.trainNetwork();
	double testErrors[] = network.testNetwork();
```

