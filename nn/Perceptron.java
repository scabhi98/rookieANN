package nn;

import java.util.function.Function;

public class Perceptron implements Runnable {
	String name;
	PerceptronLayer prev_layer;
	double activation, bias, link_weights[], delError, netInput, lastWeightCorrection[], lastBiasCorrection;
	Function <Double,Double> transferFunction;
	public Perceptron() {
		activation = bias = 0;
		link_weights = null;
		prev_layer = null;
	}
	public Perceptron(double act, double bias, double[] weights, PerceptronLayer prev_layer) {
		this.prev_layer = prev_layer; 
		activation = act;
		this.bias = bias;
		this.link_weights = weights;
	}
	/**
	 * @return the netInput
	 */
	public double getNetInput() {
		return netInput;
	}
	@Override
	public void run(){
		netInput = 0;
		for (int i = 0; i < link_weights.length; i++) 
			netInput += prev_layer.getNode(i).activation * link_weights[i];
		netInput -= bias;
		activation = (double) transferFunction.apply(netInput);
		System.out.println(this + " Activated on "+activation);
	}

	/**
	 * @param transferFunction the transferFunction to set
	 */
	public void setTransferFunction(Function<Double, Double> transferFunction) {
		this.transferFunction = transferFunction;
	}

	/**
	 * @return the lastBiasCorrection
	 */
	public double getLastBiasCorrection() {
		return lastBiasCorrection;
	}
	/**
	 * @param lastBiasCorrection the lastBiasCorrection to set
	 */
	public void setLastBiasCorrection(double lastBiasCorrection) {
		this.lastBiasCorrection = lastBiasCorrection;
	}
	/**
	 * @param lastWeightCorrection the lastWeightCorrection to set
	 */
	public void setLastWeightCorrection(double[] lastWeightCorrection) {
		this.lastWeightCorrection = lastWeightCorrection;
		for (int i = 0; i < lastWeightCorrection.length; i++) {
			this.lastWeightCorrection[i] = 0;
		}
	}
	/**
	 * @return the lastWeightCorrection
	 */
	public double[] getLastWeightCorrection() {
		return lastWeightCorrection;
	}

	/**
	 * @param delError the delError to set
	 */
	public void setDelError(double delError) {
		this.delError = delError;
	}
	/**
	 * @return the delError
	 */
	public double getDelError() {
		return delError;
	}

	@Override
	public String toString() {
		return name;
	}
	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @param link_weights the link_weights to set
	 */
	public void setLink_weights(double[] link_weights) {
		this.link_weights = link_weights;
	}
	/**
	 * @param activation the activation to set
	 */
	public void setActivation(double activation) {
		this.activation = activation;
		System.out.println(this + " is activated at: "+activation);
	}
	/**
	 * @param bias the bias to set
	 */
	public void setBias(double bias) {
		this.bias = bias;
	}
	/**
	 * @return the activation
	 */
	public double getActivation() {
		return activation;
	}
	/**
	 * @return the bias
	 */
	public double getBias() {
		return bias;
	}
	/**
	 * @return the link_weights
	 */
	public double[] getLink_weights() {
		return link_weights;
	}
	/**
	 * @param prev_layer the prev_layer to set
	 */
	public void setPrevLayer(PerceptronLayer prev_layer) {
		this.prev_layer = prev_layer;
	}
}
