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

	public Thread [] runLayer(){
		Thread [] nodeThreads = new Thread[nodes.length];
		for (int i = 0; i < nodeThreads.length; i++) {
			nodeThreads[i] = new Thread(nodes[i]);
			nodeThreads[i].start();
		}
		return nodeThreads;
	}
	/**
	 * 
	 * @throws InterruptedException
	 */
	public void waitForLayerThreads(Thread threads[]) throws InterruptedException {
		for (Thread perceptron : threads) 
			perceptron.join();
	}
	/**
	 * 
	 * @param input
	 * @throws InvalidArgumentVectorSize
	 */
	public void feedInputActivation(double []input) throws InvalidArgumentVectorSize {
		if(input.length == nodes.length){
			for (int i = 0; i < input.length; i++) 
				nodes[i].setActivation(input[i]);
		}
		else
			throw new InvalidArgumentVectorSize();
	}
	
	public void calcLayerOutput() throws InterruptedException {
		waitForLayerThreads(runLayer());
	}
	
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
	 * @throws InvalidArgumentVectorSize: If vector sizes do not matches with nodes count.
	 * @param weights is List of double[] that takes weight vector for each node
	 * @param activation is double vector to initialize activation value of each node
	 * @param bias is double vector to initialize bias value of each node.
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
