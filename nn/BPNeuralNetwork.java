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
	 * 
	 * @param layer_count
	 * @param layer_strength_vector
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
	 * 
	 * @param layer_weights
	 * @param layer_activations
	 * @param layer_biases
	 * @throws InvalidArgumentVectorSize
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
	

	public void configAllInitialValuesToRandom() throws InvalidArgumentVectorSize {
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
				perceptronLayer.init_nodes(weightList, actList, biasList, (Double x) -> {return transferFunc(x);});
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

	public double[] testNetwork() 
		throws InvalidArgumentVectorSize, InterruptedException, NoSuchLayerException, IOException 
	{
		List<double []> testInputs = dataAdapter.getTestInputList();
		List<double []> testOutputs = dataAdapter.getTestOutputList();
		double []errors = new double[testInputs.size()];
		for (int i = 0; i < testInputs.size(); i++) {
			double input[] = testInputs.get(i);
			double output[] = testOutputs.get(i);
			propagateInput(input);
			double []calcOutput = getNetworkOutput();
			for (int j = 0; j < calcOutput.length; j++) {
				dataAdapter.printColumn(j, output[j]);
				System.out.println(output[j]);
				dataAdapter.printColumn(j, calcOutput[j]);
			}
				
			System.out.println("RMS ERROR: " + (errors[i] = rmsError(output,calcOutput)));
		}
		return errors;
	}

	public double[] trainNetwork()
			throws InvalidArgumentVectorSize, InterruptedException, NoSuchLayerException, IOException {
		List<double []> trainInputs = dataAdapter.getTrainingInputList();
		List<double []> trainOutputs = dataAdapter.getTrainingOutputList();
		double []errors = new double[trainInputs.size()];
		for (int i = 0; i < trainInputs.size(); i++) {
			double input[] = trainInputs.get(i);
			double output[] = trainOutputs.get(i);
			propagateInput(input);
			
			calcAndBackPropagateError(output);
			double []calcOutput = getNetworkOutput();
			for (int j = 0; j < calcOutput.length; j++) {
				dataAdapter.printColumn(j, calcOutput[j]);
				dataAdapter.printColumn(j, output[j]);
				System.out.println(output[j]);
			}
				
			System.out.println("RMS ERROR: " + (errors[i] = rmsError(output,calcOutput)));
		}
		return errors;
	}
	/**
	 * 
	 * @return
	 */
	public double [] getNetworkOutput() {
		return OUTPUT_LAYER.getLayerOutput();
	}
	/**
	 * 
	 * @param input
	 * @throws InvalidArgumentVectorSize
	 * @throws InterruptedException
	 */
	public void propagateInput(double [] input) throws InvalidArgumentVectorSize, InterruptedException {
		INPUT_LAYER.feedInputActivation(input);
//		double [] holdOutput = layers[0].getLayerOutput();
		for(int i=1;i<layers.length;i++) {
			layers[i].calcLayerOutput();
		}
	}
	/**
	 * 
	 * @param targetOutput
	 * @return
	 * @throws InvalidArgumentVectorSize
	 * @throws NoSuchLayerException
	 */
	public double [] calcAndBackPropagateError(double []targetOutput) throws InvalidArgumentVectorSize,
			NoSuchLayerException {
		if(targetOutput.length != OUTPUT_LAYER.getNodesCount())
			throw new InvalidArgumentVectorSize();
		double outputError[] = new double[targetOutput.length];
		/*Calculation of Error at Output Layer*/
		for (int i = 0; i < targetOutput.length; i++) {
		   outputError[i]	= calcDelErrorAtOutputLayer(i, OUTPUT_LAYER, targetOutput[i]);
//		   Perceptron node = OUTPUT_LAYER.getNode(i);
//		   double biasCorr = errorToBiasDeriv(i,OUTPUT_LAYER);
//			biasCorr = biasCorr *learning_rate - momentum * node.getBias();
//			node.setBias(node.getBias() - biasCorr);
//			node. -= learning_rate * corr - momentum *node.getLink_weights()[k];
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
					System.out.println("Weight of " + node + " at " + k + " is corrected to: " + node.getLink_weights()[k]);
				}
				double biasCorr = errorToBiasDeriv(j,workingLayer);
				biasCorr = biasCorr *learning_rate + momentum * node.getLastBiasCorrection();
				node.setBias(node.getBias() - biasCorr);
				node.setLastBiasCorrection(biasCorr);
				System.out.println("Bias of "+node+" is corrected to: "+node.getBias());
			}
		}
		return outputError;
	}
}
