package nn;

public interface BPCalculus {

    abstract double transferFunc(double x);
   
    abstract double transferDeriv(double x) ;
   
    default double errorToOutputDeriv(int node_pos, PerceptronLayer layer , double targetValue){
        double v = 2*(layer.getNode(node_pos).getActivation() - targetValue);
        layer.getNode(node_pos).setDelError(v);
        System.out.println("Error Calculated at "+ layer.getNode(node_pos)+ " is "+v);
        return v;
    }
   
    default double inputToWeightDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer)
            throws NoSuchLayerException {
        if (layer.getPrevLayer() == null)
            throw new NoSuchLayerException();
        return layer.getPrevLayer().getNode(dest_node_pos).getActivation();
    }
   
    default double activationToInputDeriv(int node_pos, PerceptronLayer layer){
        return transferDeriv(layer.getNode(node_pos).getActivation());
    }
   
    default double inputToActivationDeriv(int host_node_pos, int dest_node_pos, PerceptronLayer layer){
        return layer.getNode(host_node_pos).getLink_weights()[dest_node_pos];
    }
   
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
    	if(layer.getNextLayer() == null) {
    		Perceptron node = layer.getNode(host_node_pos);
    		val = transferDeriv(node.getActivation()) * node.getDelError();
    	}
    	else {
    		PerceptronLayer layer2 = layer.getNextLayer();
    		for(int i=0; i< layer2.getNodesCount();i++){
                Perceptron node = layer2.getNode(i);
                val += transferDeriv(node.getNetInput()) * node.getDelError();
            }
    	}
    	return val;
    }

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
        System.out.println("Error Calculated At "+ layer.getNode(host_node_pos)+" to "+dest_node_pos+" is "+ val);
        return val;
    }
}