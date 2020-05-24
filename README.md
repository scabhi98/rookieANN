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
class DLBCLAdapter extends DataAdapter {

	public DLBCLAdapter(final InputStream inputStream, final OutputStream outputStream, final int inputColumnWidth, final int outputColumnWidth) {
		super(inputStream, outputStream, inputColumnWidth, outputColumnWidth);
	}

	@Override
	public String formatOutput(double []output){
		if((output[0] > 0.7 && output[1] < 0.2 ) || output[0] > 0.9)
			return " FL ";
		if((output[0] < 0.2 && output[1] > 0.7) || output[1] > 0.9 )
			return " DLBCL ";
		return " Confused ";
	}

	@Override
	public void normalizeInputRow(String row, double[] inputs, double[] outputs) {
		Scanner sc = new Scanner(row);
		sc.useDelimiter(",");
		
		for (int i = 0; i < inputs.length; i++) 
			inputs[i] = Double.valueOf(sc.next());
		
		switch(sc.next()){
			case "DLBCL":
				outputs[0] = 0;
				outputs[1] = 1;
				break;
			case "FL":
				outputs[0] = 1;
				outputs[1] = 0;
				break;
		}
		
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
	DataAdapter dAdapter = new DLBCLAdapter(new FileInputStream(inpFile), new FileOutputStream(oFile), 12, 1);
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

