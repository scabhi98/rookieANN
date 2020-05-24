import java.io.InputStream;
import java.io.OutputStream;
import java.util.Scanner;

import nn.DataAdapter;

class DLBCLAdapter extends DataAdapter {

	public DLBCLAdapter(final InputStream inputStream, final OutputStream outputStream, final int inputColumnWidth, final int outputColumnWidth) {
		super(inputStream, outputStream, inputColumnWidth, outputColumnWidth);
	}

	@Override
	public String formatOutput(double []output){
		if((output[0] > 0.7 && output[1] < 0.2 ) || output[0] > 0.9)
			return " FL ";
		if((output[0] < 0.2 && output[1] > 0.7) || output[1] > 0.9 )
			return " DLBCL ";
		return " Confused ";
	}

	double min_max_normalization(double value, double min, double max){
		return (value - min) / (max - min);
	}

	@Override
	public void normalizeInputRow(String row, double[] inputs, double[] outputs) {
		Scanner sc = new Scanner(row);
		sc.useDelimiter(",");
		
		for (int i = 0; i < inputs.length; i++) 
			inputs[i] = Double.valueOf(sc.next());
		
		switch(sc.next()){
			case "DLBCL":
				outputs[0] = 0;
				outputs[1] = 1;
				break;
			case "FL":
				outputs[0] = 1;
				outputs[1] = 0;
				break;
		}
		
		sc.close();
	}
	
}