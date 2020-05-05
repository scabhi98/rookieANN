package nn;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public abstract class DataAdapter {
    List<double[]> inputs;
    List<double[]> outputs;
    int inputColumnWidth, outputColumnWidth, restrictedDataSize;
    InputStream inputSourceStream;
    OutputStream outputTargetStream;
    float partitionRatio;

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

    public abstract void normalizeInputRow(String row, double inputs[], double outputs[]);

    public abstract String formattedOutputColumn(int columnPosition, double calcOutput);

    protected void generateLists() {
        int rowWidth = inputColumnWidth + outputColumnWidth;
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
     * @param restrictedDataSize the restrictedDataSize to set
     */
    public void setRestrictedDataSize(int restrictedDataSize) {
        this.restrictedDataSize = restrictedDataSize;
    }

    public void printColumn(int columnPosition, double calcOutput) throws IOException {
        outputTargetStream.write(("\n  "+formattedOutputColumn(columnPosition, calcOutput)).getBytes());
    }

    public List<double[]> getTestInputList() {
        int last = restrictedDataSize - 1;
        int start =(int) (((float) last) * partitionRatio);
        return inputs.subList(start, last);
    }
    public List<double[]> getTestOutputList() {
        int last = restrictedDataSize - 1;
        int start =(int) (((float) last) * partitionRatio);
        return outputs.subList(start, last);
    }

    public List<double[]> getTrainingInputList() {
        int last =(int) (((float) restrictedDataSize) * partitionRatio);
        return inputs.subList(0, last);
    }
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
}