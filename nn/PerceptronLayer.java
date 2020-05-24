package nn;

import java.util.List;
import java.util.function.Function;

public class PerceptronLayer {
	
	Perceptron nodes[];
	String LayerID;
	
	PerceptronLayer prevLayer, nextLayer;
	public PerceptronLayer(){
		nodes = null;
		prevLayer = nextLayer = null;
		LayerID = null;
	}
	
	public PerceptronLayer(int strength){
		nodes = new Perceptron[strength];
		for (int i = 0; i < nodes.length; i++) {
			nodes[i] = new Perceptron();
		}
	    nextLayer  =  prevLayer = null;
	}
	/**
	 * @param layer to connect the preceding layer in network
	 * */
	public void connectLayer(PerceptronLayer layer) {
		this.nextLayer = layer;
		layer.prevLayer = this;
	}

	/**
	 * @param layerID the layerID to set
	 */
	public void setLayerID(String layerID) {
		LayerID = layerID;
	}

	@Override
	public String toString() {
		return LayerID;
	}

	/**
	 * @return the nextLayer
	 */
	public PerceptronLayer getNextLayer() {
		return nextLayer;
	}
	/**
	 * @return the prevLayer
	 */
	public PerceptronLayer getPrevLayer() {
		return prevLayer;
	}

	/**
	 * @param position of node in the layer
	 * @return the node of the layer
	 */

	public Perceptron getNode(int position) {
		return nodes[position];
	}

	Thread [] runLayer(){
		Thread [] nodeThreads = new Thread[nodes.length];
		for (int i = 0; i < nodeThreads.length; i++) {
			nodeThreads[i] = new Thread(nodes[i]);
			nodeThreads[i].start();
		}
		return nodeThreads;
	}
	void waitForLayerThreads(Thread threads[]) throws InterruptedException {
		for (Thread perceptron : threads) 
			perceptron.join();
	}
	/**
	 * Feeds the input values to the perceptrons in this layer.
	 * @param input vector of input values.
	 * @throws InvalidArgumentVectorSize on mismatch size of the parameter and layer nodes count.
	 */
	public void feedInputActivation(double []input) throws InvalidArgumentVectorSize {
		if(input.length == nodes.length){
			for (int i = 0; i < input.length; i++) 
				nodes[i].setActivation(input[i]);
		}
		else
			throw new InvalidArgumentVectorSize();
	}
	/**
	 * Invokes all the perceptrons in the layer and wait for their completion to gain the final output
	 * @throws InterruptedException on interruption of any running perceptron in the layer.
	 */
	
	public void calcLayerOutput() throws InterruptedException {
		waitForLayerThreads(runLayer());
	}
	/**
	 * Retrieves the activation i.e. output value of each node as a vector for this layer.
	 * @return vector of floating point values representing output values of each node of this layer.
	 */
	public double [] getLayerOutput() {
		double [] output = new double[nodes.length];
		for (int i = 0; i < output.length; i++) 
			output[i] = nodes[i].getActivation();
		return output;
	}

	/**
	 * @param count number of nodes to allocate in this layer.
	 */

	public void allocNodes(int count){
		nodes = new Perceptron[count];
	}

	/**
	 * @return the nodes
	 */
	public int getNodesCount() {
		return nodes.length;
	}
	
	/**
	 * @throws InvalidArgumentVectorSize If vector sizes do not matches with nodes count.
	 * @param weights is List of double[] that takes weight vector for each node
	 * @param activation is double vector to initialize activation value of each node
	 * @param bias is double vector to initialize bias value of each node.
	 * @param transferFunction is transfer Function definition for each node.
	 * @throws InvalidArgumentVectorSize if any of the argument vector size differs with the corresponding layer configuration.
	 * */
	
	public void init_nodes(List<double []> weights, double activation[], double bias[], Function<Double,Double> transferFunction) 
	throws InvalidArgumentVectorSize {
		if (weights.size() == nodes.length && activation.length == nodes.length && bias.length == nodes.length) {
			for(int i=0; i<nodes.length; i++) {
				double [] w = weights.get(i);
				if(w.length == prevLayer.nodes.length) 
					nodes[i].setLink_weights(w);
				else
					throw new InvalidArgumentVectorSize();
				nodes[i].setName(LayerID+"/Node "+i);
				nodes[i].setPrevLayer(prevLayer);
				nodes[i].setActivation(activation[i]);
				nodes[i].setBias(bias[i]);
				nodes[i].setTransferFunction(transferFunction);	
				nodes[i].setLastWeightCorrection(new double[w.length]);
				nodes[i].setLastBiasCorrection(0);
			}
		}
		else
			throw new InvalidArgumentVectorSize();
	}
}
