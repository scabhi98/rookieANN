package nn;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public abstract class BPNeuralNetwork implements BPCalculus{
	PerceptronLayer layers[];
	double learning_rate, momentum;
	PerceptronLayer INPUT_LAYER;
	PerceptronLayer OUTPUT_LAYER;
	DataAdapter dataAdapter;
	/**
	 * Constructs a Backpropagation based Neural Network
	 * sets momentum to 0 and learning rate to 0.5.
	 * @param layer_strength_vector array of integers indicating number of nodes at each layer
	 * in order. Must contains at least two elements i.e. Input width and output width.
	 */
	public BPNeuralNetwork(int layer_strength_vector[]) {
		int layer_count = layer_strength_vector.length;
		layers = new PerceptronLayer[layer_count];
		for (int i = 0; i < layer_count; i++) 
			layers[i] = new PerceptronLayer(layer_strength_vector[i]);
		for(int i = 0; i< layer_count; i++) {
			// layers[i].allocNodes(layer_strength_vector[i]);
			layers[i].setLayerID("Layer "+i);
			if(i < layer_count - 1)
				layers[i].connectLayer(layers[i+1]);
		}
		INPUT_LAYER = layers[0];
		OUTPUT_LAYER = layers[layer_count - 1];
		momentum = 0;
		learning_rate = 0.5;
	}

	/**
	 * @param dataAdapter the dataAdapter to set
	 */
	public void setDataAdapter(DataAdapter dataAdapter) {
		this.dataAdapter = dataAdapter;
	}
	/**
	 * @param learning_rate the learning_rate to set
	 */
	public void setLearning_rate(double learning_rate) {
		this.learning_rate = learning_rate;
	}
	/**
	 * @return the learning_rate
	 */
	public double getLearning_rate() {
		return learning_rate;
	}
	/**
	 * @param momentum the momentum to set
	 */
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
	/**
	 * @return the momentum
	 */
	public double getMomentum() {
		return momentum;
	}
	/**
	 * Configure network weights and biases to some predefined values.
	 * @param layer_weights List of layer weights where each element is a List of link weights of each node in the layer.
	 * @param layer_activations List of initial activation values for network where each element is an array of values for each layer nodes.
	 * @param layer_biases List of initial bias values for network where each element is an array of values for each layer nodes.
	 * @throws InvalidArgumentVectorSize if any of the argument vector size differs with the corresponding network configuration.
	 */
	public void configureLayers(List<List <double []>> layer_weights, List <double []> layer_activations, List<double []> layer_biases) throws InvalidArgumentVectorSize {
		if(layer_weights.size() != layers.length || layer_activations.size() != layers.length || layer_biases.size() != layers.length) {
			throw new InvalidArgumentVectorSize();
		}
		else {
			for(int i=0;i<this.layers.length; i++) {
				layers[i].init_nodes(layer_weights.get(i), layer_activations.get(i), layer_biases.get(i), (Double x) -> {return transferFunc(x);} );
			}
		}
	}
	/**
	 * Sets all initial weights and bias of the network nodes to random values between 0 to 1
	 */

	public void configAllInitialValuesToRandom() {
		Random r = new Random(System.currentTimeMillis());
		for (PerceptronLayer perceptronLayer : layers) {
			if (perceptronLayer != INPUT_LAYER) {
				List<double []> weightList = new ArrayList<>();
				double w[], actList[] = null, biasList[] = null;
				actList = new double[perceptronLayer.getNodesCount()];
				biasList = new double[perceptronLayer.getNodesCount()];
				for(int i=0; i<perceptronLayer.getNodesCount();i++){
					w = new double[perceptronLayer.getPrevLayer().getNodesCount()];
					for (int j = 0; j<w.length; j++) 
						w[j] = r.nextDouble();
					weightList.add(w);
					actList[i] = r.nextDouble();
					biasList[i] = r.nextDouble()/2;
				}
				try{
					if(perceptronLayer != OUTPUT_LAYER)
						perceptronLayer.init_nodes(weightList, actList, biasList, (Double x) -> {return transferFunc(x);});
					
					else
						perceptronLayer.init_nodes(weightList, actList, biasList, (Double x) -> {return transferFuncAtOutput(x);});
				}catch(InvalidArgumentVectorSize e){
					e.printStackTrace();
				}
			}
			else{
				for(int i=0; i<perceptronLayer.getNodesCount();i++){
					Perceptron node = perceptronLayer.getNode(i);
					node.setName("InputLayer/Node "+i);
					node.setActivation(r.nextDouble());
					node.setBias(r.nextDouble()/2);
					node.setLink_weights(null);
					node.setPrevLayer(null);
					node.setTransferFunction((Double x) -> {return transferFunc(x);});
				}
			}			
		}
	}

	/**
	 * To start testing of network targetting the test set defined by the DataAdapter object passed.
	 * @return array of double values each indicating rms error between actual and predicted values.
	 * @throws InvalidArgumentVectorSize on encounter of non-matching length between network input nodes and the input vector size stored in DataAdapter
	 * @throws InterruptedException	on interrupton of running Perceptron Thread.
	 * @throws IOException if output file stream does not work properly.
	 */

	public double[] testNetwork() 
		throws InvalidArgumentVectorSize, InterruptedException, IOException 
	{
		List<double []> testInputs = dataAdapter.getTestInputList();
		List<double []> testOutputs = dataAdapter.getTestOutputList();
		double []errors = new double[testInputs.size()];
		for (int i = 0; i < testInputs.size(); i++) {
			double input[] = testInputs.get(i);
			double output[] = testOutputs.get(i);
			propagateInput(input);
			double []calcOutput = getNetworkOutput();
			dataAdapter.printText("\nActual\n => \t");
			dataAdapter.printOutput(output);
			dataAdapter.printText("\nPredicted\n => \t");
			dataAdapter.printOutput(calcOutput);
			dataAdapter.printText("\n -----------------------\n");
				
			System.out.println("RMS ERROR: " + (errors[i] = rmsError(output,calcOutput)));
		}
		return errors;
	}

	/**
	 * Initiates training of networking by feeding the training partition of dataset
	 * from DataAdapter.
	 * 
	 * @return array of rms errors occurred at last stage of retraining by each input. Length is equal to Training data set.
	 * @throws InvalidArgumentVectorSize on encounter of mismatch length of input or output vector size and the network input or output width.
	 * @throws InterruptedException on interruption of running Perceptron
	 * @throws IOException on error occurred during writing the output to output file stream attached to DataAdapter.
	 * @throws NoSuchLayerException	occurs when network access some layer undefined.
	 */

	public double[] trainNetwork()
			throws InvalidArgumentVectorSize, InterruptedException, IOException, NoSuchLayerException {
		List<double []> trainInputs = dataAdapter.getTrainingInputList();
		List<double []> trainOutputs = dataAdapter.getTrainingOutputList();
		double []errors = new double[trainInputs.size()];
		int retrain = 128;
		for (int i = 0; i < trainInputs.size(); i++) {
			double input[] = trainInputs.get(i);
			double output[] = trainOutputs.get(i);
			double []calcOutput;
			int MAXREPEAT = 5000;
			do{
				propagateInput(input);
				calcAndBackPropagateError(output);
				calcOutput = getNetworkOutput();
				errors[i] = rmsError(output,calcOutput);
				dataAdapter.printText("\nActual\n");
				dataAdapter.printOutput(output);
				dataAdapter.printText("\nPredicted\n");
				dataAdapter.printOutput(calcOutput);
				dataAdapter.printText("\n ----------"+i+" "+retrain+"-------------\n");
				System.out.print("\n ----------"+i+" "+retrain+"-------------\n");
					
				System.out.println("RMS ERROR: " + errors[i]);
				if(MAXREPEAT-- == 0)
					break;
			}while(errors[i] > 1 || (errors[i]> 0.1 && retrain-- > 0));
			retrain = 128;				
		}
		return errors;
	}
	/**
	 * Retrieves the network output at any stage.
	 * @return the output values of each output node as an array.
	 */
	public double [] getNetworkOutput() {
		return OUTPUT_LAYER.getLayerOutput();
	}
	/**
	 * Run network against an input value. It assigns the the input values to input layer nodes first
	 * then propagates them through the network by starting execution of each layer in order.
	 * @param input input vector that is supposed to be feeded to the network.
	 * @throws InvalidArgumentVectorSize on encounter of unequal input vector size with the network width.
	 * @throws InterruptedException	on interruption of any running Perceptron node.
	 */
	public void propagateInput(double [] input) throws InvalidArgumentVectorSize, InterruptedException {
		INPUT_LAYER.feedInputActivation(input);
//		double [] holdOutput = layers[0].getLayerOutput();
		for(int i=1;i<layers.length;i++) {
			layers[i].calcLayerOutput();
		}
	}
	/**
	 * To calculate error occurred at the output node with actual value and then propagate them backwards
	 * by calibration of link weights and node bias values using gradient decent calculus.
	 * @param targetOutput the actual output for the last feeded input.
	 * @return errors at each node in the output layer.
	 * @throws InvalidArgumentVectorSize on encounter of invalid size of @param targetOutput with the output width of network.
	 * @throws NoSuchLayerException if network configuration is invalid and an undefined layer is accessed.
	 */
	public double [] calcAndBackPropagateError(double []targetOutput) throws InvalidArgumentVectorSize,
			NoSuchLayerException {
		if(targetOutput.length != OUTPUT_LAYER.getNodesCount())
			throw new InvalidArgumentVectorSize();
		double outputError[] = new double[targetOutput.length];
		/*Calculation of Error at Output Layer*/
		for (int i = 0; i < targetOutput.length; i++) {
		   outputError[i]	= calcDelErrorAtOutputLayer(i, OUTPUT_LAYER, targetOutput[i]);
		}
		for(int i = layers.length - 1; i > 0; i--){
			PerceptronLayer workingLayer = layers[i];
			for(int j = 0; j<workingLayer.getNodesCount(); j++){
				Perceptron node =  workingLayer.getNode(j);
				PerceptronLayer prevLayer = workingLayer.getPrevLayer();
				for (int k =0; k<prevLayer.getNodesCount(); k++){
					double corr = errorToWeightDeriv(j, k, workingLayer);
					node.getLink_weights()[k] -= (corr  = learning_rate * corr * prevLayer.getNode(k).getActivation() +
					momentum * node.getLastWeightCorrection()[k]);
					node.getLastWeightCorrection()[k] = corr;
					// System.out.println("Weight of " + node + " at " + k + " is corrected to: " + node.getLink_weights()[k]);
				}
				double biasCorr = errorToBiasDeriv(j,workingLayer);
				biasCorr = biasCorr * learning_rate + momentum * node.getLastBiasCorrection();
				node.setBias(node.getBias() - biasCorr);
				node.setLastBiasCorrection(biasCorr);
				// System.out.println("Bias of "+node+" is corrected to: "+node.getBias());
			}
		}
		return outputError;
	}
}
