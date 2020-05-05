package nn;

import java.io.File;
import java.util.List;
import java.util.Scanner;

public abstract class DataAdapter {
    List<double[]> inputs;
    List<double[]> outputs;
    int inputColumnWidth, outputColumnWidth;
    File inputFile;
    float partitionRatio;

    public DataAdapter(File inpFile, int inputColumnWidth, int outputColumnWidth) {
        this.inputColumnWidth = inputColumnWidth;
        this.outputColumnWidth = outputColumnWidth;
        this.inputFile = inpFile;
        partitionRatio = (float) 0.65;
    }

    abstract double normalizeInputColumn(int columnPosition, String value);

    protected void generateLists() {
        int rowWidth = inputColumnWidth + outputColumnWidth;
        try {
            Scanner sc = new Scanner(inputFile);
            sc.useDelimiter(",");
            while(sc.hasNext()){
                double inp[] = new double[inputColumnWidth], op[] = new double[outputColumnWidth];
                for (int i = 0; i < rowWidth; i++) {
                    if(i<inputColumnWidth)
                        inp[i] = normalizeInputColumn(i, sc.next());
                    else
                        op[i - inputColumnWidth] = normalizeInputColumn(i, sc.next());
                }
                inputs.add(inp);
                outputs.add(op);                
            }
            sc.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    public List<double[]> getTestInputList() {
        int last = inputs.size() - 1;
        int start =(int) (((float) last) * partitionRatio);
        return inputs.subList(start, last);
    }
    public List<double[]> getTestOutputList() {
        int last = outputs.size() - 1;
        int start =(int) (((float) last) * partitionRatio);
        return outputs.subList(start, last);
    }

    /**
     * @param inputFile the inputFile to set
     */
    public void setInputFile(File inputFile) {
        this.inputFile = inputFile;
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