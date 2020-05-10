package nn;

public interface BPCalculus {

    abstract double transferFunc(double x);
   
    abstract double transferDeriv(double x) ;
   
    default double calcDelErrorAtOutputLayer(int node_pos, PerceptronLayer layer , double targetValue){
        double x = layer.getNode(node_pos).getActivation();
        double v = (x - targetValue) * x * (1 -  x);
        layer.getNode(node_pos).setDelError(v);
        System.out.println("Del Error Function Calculated at "+ layer.getNode(node_pos)+ " is "+v);
        return v;
    }
   
    // default double inputToWeightDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer)
    //         throws NoSuchLayerException {
    //     if (layer.getPrevLayer() == null)
    //         throw new NoSuchLayerException();
    //     return layer.getPrevLayer().getNode(dest_node_pos).getActivation();
    // }
   
    // default double activationToInputDeriv(int node_pos, PerceptronLayer layer){
    //     return transferDeriv(layer.getNode(node_pos).getActivation());
    // }
   
    // default double inputToActivationDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer){
    //     return layer.getNode(host_node_pos).getLink_weights()[dest_node_pos];
    // }
   
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
    
    default double errorToBiasDeriv(int host_node_pos, PerceptronLayer layer) {
        double val=0;
        Perceptron node = layer.getNode(host_node_pos);
    	if(layer.getNextLayer() == null) {
    		val = node.getDelError();
    	}
    	else {
    		PerceptronLayer layer2 = layer.getNextLayer();
    		for(int i=0; i< layer2.getNodesCount();i++){
                Perceptron node2 = layer2.getNode(i);
                val +=  node2.getDelError();
            }
    	}
    	return transferDeriv(node.getNetInput()) * val;
    }

    default double errorToWeightDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer)
            throws NoSuchLayerException {
        double val = 0;
        if(layer.getNextLayer() == null){
            val = 
                // inputToWeightDeriv(host_node_pos, dest_node_pos, layer)*
                // activationToInputDeriv(host_node_pos, layer)*
                layer.getNode(host_node_pos).getDelError();

        }
        else{
            val = 
                // inputToWeightDeriv(host_node_pos, dest_node_pos, layer)*
                // activationToInputDeriv(host_node_pos, layer)*
                // errorToActivationDeriv(host_node_pos, layer);                
                delError(host_node_pos, layer); 
        }
        System.out.println("Error Calculated At "+ layer.getNode(host_node_pos)+" to "+dest_node_pos+" is "+ val);
        return val;
    }

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

    default double delError(int pos, PerceptronLayer layer) throws NoSuchLayerException {
        if (layer.getNextLayer() == null) {
            throw new NoSuchLayerException();
        } else {
            double x = layer.getNode(pos).getActivation(), sum = 0;
            PerceptronLayer layer2 = layer.getNextLayer();
            for(int i=0; i< layer2.getNodesCount();i++){
                Perceptron node = layer2.getNode(i);
                sum += node.getLink_weights()[pos] * node.getDelError();
            }
            double err = sum * x * (1-x);
            layer.getNode(pos).setDelError(err);
            return err;
        }
    }
}