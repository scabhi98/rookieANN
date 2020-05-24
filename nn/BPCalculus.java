package nn;

public interface BPCalculus {

    /**
     * Abstract method that defines transfer function to be used by an Perceptron.
     * @param x input value
     * @return output value of the function.
     */
    abstract double transferFunc(double x);
   /**
    * Abstract method that defines derivative of the defined transfer function for Perceptron.
    * @param x input value
    * @return value returned by the derivative function.
    */
    abstract double transferDeriv(double x) ;

    /**
     * If any different transfer function is supposed to be used at output layer. Default definition is the common transfer function.
     * @param x input value.
     * @return output value.
     */

    default double transferFuncAtOutput(double x) {
        return transferFunc(x);
    }
    /**
     * Derivative function of output layer transfer function if different transfer function is used than common.
     * @param x input value.
     * @return value returned by the derivative funtion.
     */
    default double transferDerivAtOutput(double x) {
        return transferDeriv(x);
    }

    /**
     * Calculates the delta error function at output layer node.
     * @param node_pos node position of working node.
     * @param layer Output layer needed to be passed.
     * @param targetValue target value the node is supposed to give.
     * @return error calculated at specified output node.
     */
   
    default double calcDelErrorAtOutputLayer(int node_pos, PerceptronLayer layer , double targetValue){
        double x = layer.getNode(node_pos).getActivation();
        double v = 2*(x - targetValue);
        layer.getNode(node_pos).setDelError(v);
        // System.out.println("Del Error Function Calculated at "+ layer.getNode(node_pos)+ " is "+v);
        return v;
    }
   /**
    * To calculate the partial derivative of input w.r.t. weight of link between node defined by host_node_position to
    * the node specified by the dest_node_pos in the previous layer.
    * Default defined by Gradient Decent calculus.
    * @param host_node_pos node position of current node.
    * @param dest_node_pos node position of linked node in previous layer.
    * @param layer in which the current node is.
    * @return value of partial derivative calculation.
    * @throws NoSuchLayerException if layer is Input layer as there is no linked previous layer of it.
    */
    default double inputToWeightDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer)
            throws NoSuchLayerException {
        if (layer.getPrevLayer() == null)
            throw new NoSuchLayerException();
        return layer.getPrevLayer().getNode(dest_node_pos).getActivation();
    }
   /**
    * To calculate the partial derivative of activation of node w.r.t. net input received.
    * Default defined by Gradient Decent calculus.
    * @param node_pos specifies node in the layer.
    * @param layer specifies the layer.
    * @return value determined by the derivative calculation.
    */
    default double activationToInputDeriv(int node_pos, PerceptronLayer layer){
        if(layer.getNextLayer() == null)
            return transferDerivAtOutput(layer.getNode(node_pos).getActivation());
        else
            return transferDeriv(layer.getNode(node_pos).getActivation());
    }
    /**
     * To calculate the partial derivative of error w.r.t. activation of a node defined by host_node_position in a layer.
     * Default defined by Gradient Decent calculus.
     * @param host_node_pos specifies the node.
     * @param layer specifies the working layer.
     * @return derivative calculation final value.
     */
    default double errorToActivationDeriv(int host_node_pos, PerceptronLayer layer){
        if(layer.getNextLayer() == null)
            throw new NullPointerException();
        double sum = 0;
        PerceptronLayer layer2 = layer.getNextLayer();
        for(int i=0; i< layer2.getNodesCount();i++){
            Perceptron node = layer2.getNode(i);
            sum += node.getLink_weights()[host_node_pos] * transferDeriv(node.getNetInput()) * node.getDelError();
        }
        layer.getNode(host_node_pos).setDelError(sum);
        return host_node_pos;
    }
    /**
     * To calculate the partial derivative of error w.r.t. bias of node defined by host_node_position in a layer.
     * Default defined by Gradient Decent calculus.
     * @param host_node_pos specifies the node.
     * @param layer specifies the layer.
     * @return value of partial derivative calculation.
     */
    default double errorToBiasDeriv(int host_node_pos, PerceptronLayer layer) {
        double val=0;
        Perceptron node = layer.getNode(host_node_pos);
    	if(layer.getNextLayer() == null) {
    		val = transferDeriv(node.getNetInput()) * node.getDelError();
    	}
    	else {
    		PerceptronLayer layer2 = layer.getNextLayer();
    		for(int i=0; i< layer2.getNodesCount();i++){
                Perceptron node2 = layer2.getNode(i);
                val +=  transferDeriv(node.getNetInput()) * node2.getDelError();
            }
    	}
    	return  val;
    }
    /**
     * To calculate the partial derivative of error w.r.t. weight of link between node defined by host_node_position to
     * the node specified by the dest_node_pos in the previous layer.
     * Default defined by Gradient Decent calculus.
     * @param host_node_pos node position of current node.
     * @param dest_node_pos node position of linked node in previous layer.
     * @param layer in which the current node is.
     * @return value of partial derivative calculation.
     * @throws NoSuchLayerException if layer is Input layer as there is no linked previous layer of it.
     */

    default double errorToWeightDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer)
            throws NoSuchLayerException {
        double val = 0;
        if(layer.getNextLayer() == null){
            val = 
                inputToWeightDeriv(host_node_pos, dest_node_pos, layer)*
                activationToInputDeriv(host_node_pos, layer)*
                layer.getNode(host_node_pos).getDelError();

        }
        else{
            val = 
                inputToWeightDeriv(host_node_pos, dest_node_pos, layer)*
                activationToInputDeriv(host_node_pos, layer)*
                errorToActivationDeriv(host_node_pos, layer);                
        }
        return val;
    }
    /**
     * Calculates the rms Error between values of two floating point vectors.
     * @param actual value vector
     * @param calculated value vector
     * @return rms error calculated
     * @throws InvalidArgumentVectorSize on mismatch of argument sizes.
     */
    default double rmsError(double actual[], double calculated[]) throws InvalidArgumentVectorSize {
        double sum =0;
        if(actual.length != calculated.length)
            throw new InvalidArgumentVectorSize();
        for (int i = 0; i < calculated.length; i++) {
            double v = (actual[i] - calculated[i]);
            sum += v*v;
        }
        return Math.sqrt(sum);
    }
}