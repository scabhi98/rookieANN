package nn;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
/**
 * Abstract class that defines structure of DataAdapter which will abstracts the Neural Network input and output
 * formats from the format and representation of actual dataset feeded to the network.
 * 
 * One must extend this class to Personlized DataAdapter version where it can do preprocessing tasks on dataset
 * formatting of the inputs of dataset and outputs of network.
 */

public abstract class DataAdapter {
    List<double[]> inputs;
    List<double[]> outputs;
    int inputColumnWidth, outputColumnWidth, restrictedDataSize;
    InputStream inputSourceStream;
    OutputStream outputTargetStream;
    float partitionRatio;
    /**
     * Constructs an DataAdapter object with following parameters.
     * @param inputStream from which the dataset elements are supposed to be scanned.
     * @param outputStream on which output of network is supposed to be printed.
     * @param inputColumnWidth defines number of columns for inputs in one scanned row of dataset.
     * @param outputColumnWidth defines number of columns for outputs in one scanned row of dataset.
     */
    public DataAdapter(InputStream inputStream, OutputStream outputStream, int inputColumnWidth,
            int outputColumnWidth) {
        this.inputColumnWidth = inputColumnWidth;
        this.outputColumnWidth = outputColumnWidth;
        this.inputSourceStream = inputStream;
        this.outputTargetStream = outputStream;
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
        partitionRatio = (float) 0.65;
        generateLists();
        restrictedDataSize = inputs.size();
    }
    /**
     * Abstract method must be defined to format and normalize the input variables and store them to input and output vector.
     * @param row String variable provides current scanned row.
     * @param inputs input vector that will be set to normalized numeric values extracted from row.
     * @param outputs output vector that will be set to formatted numeric values extracted from row.
     */

    public abstract void normalizeInputRow(String row, double inputs[], double outputs[]);

    protected void generateLists() {
        try {
            Scanner sc = new Scanner(inputSourceStream);
            while (sc.hasNextLine()) {
                double inp[] = new double[inputColumnWidth], op[] = new double[outputColumnWidth];
                normalizeInputRow(sc.nextLine(), inp, op);
                inputs.add(inp);
                outputs.add(op);
            }
            sc.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * Restricts large Dataset to first few elements specified by the parameter.
     * @param restrictedDataSize the restrictedDataSize to set
     */
    public void setRestrictedDataSize(int restrictedDataSize) {
        this.restrictedDataSize = restrictedDataSize;
    }

    /**
     * Prints Text outputs to outputStream attached to DataAdapter 
     * @param textOutput to be printed on outputstream
     * @throws IOException on error while accessing the outputStream.
     */

    public void printText(String textOutput) throws IOException {
        outputTargetStream.write(textOutput.getBytes());
    }

    
    /**
     * @return List of input vectors for testing
     */
    public List<double[]> getTestInputList() {
        int last = restrictedDataSize - 1;
        int start =(int) (((float) last) * partitionRatio);
        return inputs.subList(start, last);
    }
    /**
     * @return List of output vectors for testing.
     */
    public List<double[]> getTestOutputList() {
        int last = restrictedDataSize - 1;
        int start =(int) (((float) last) * partitionRatio);
        return outputs.subList(start, last);
    }
    /**
     * @return List of input vectors for training.
     */
    public List<double[]> getTrainingInputList() {
        int last =(int) (((float) restrictedDataSize) * partitionRatio);
        return inputs.subList(0, last);
    }
    /**
     * @return List of output vectors for training.
     */
    public List<double[]> getTrainingOutputList() {
        int last =(int) (((float)restrictedDataSize) * partitionRatio);
        return outputs.subList(0, last);
    }
    /**
     * @param inputSourceStream the inputSourceStream to set
     */
    public void setInputSourceStream(InputStream inputSourceStream) {
        this.inputSourceStream = inputSourceStream;
        generateLists();
    }

    /**
     * @param outputTargetStream the outputTargetStream to set
     */
    public void setOutputTargetStream(OutputStream outputTargetStream) {
        this.outputTargetStream = outputTargetStream;
    }
    /**
     * @param partitionRatio the partitionRatio to set
     */
    public void setPartitionRatio(float partitionRatio) {
        this.partitionRatio = partitionRatio;
    }
    /**
     * @return the inputColumnWidth
     */
    public int getInputColumnWidth() {
        return inputColumnWidth;
    }
    /**
     * @return the outputColumnWidth
     */
    public int getOutputColumnWidth() {
        return outputColumnWidth;
    }
    /**
     * @return the inputs
     */
    public List<double[]> getInputs() {
        return inputs;
    }
    /**
     * @return the outputs
     */
    public List<double[]> getOutputs() {
        return outputs;
    }
    /**
     * Formats the output vector to Output String.
     * @param output vector may be actual or predicted.
     * @return String value for the prediction result.
     */

    abstract public String formatOutput(double []output);

    /**
     * Prints output vector to the outputstream attached.
     * @param calcOutput output vector containing floating point values.
     * @throws IOException on error while accessing the output stream.
     */
	public void printOutput(double[] calcOutput) throws IOException {
        outputTargetStream.write(formatOutput(calcOutput).getBytes());
    }
}