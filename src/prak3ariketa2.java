import java.io.FileWriter;
import java.io.PrintWriter;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class prak3ariketa2 {
	public static void main(String[] args) throws Exception {
		
		Classifier nbM = (Classifier) SerializationHelper.read(args[0]);
		
		DataSource source = new DataSource(args[1]);
		Instances data = source.getDataSet();
		if(data.classIndex() == -1) {
			data.setClassIndex(data.numAttributes() - 1);
		}
		FileWriter fw = new FileWriter(args[2]);
        PrintWriter pw = new PrintWriter(fw);
        
		System.out.println("ID | Iragarritako Klasea");
        System.out.println("-----------------------");
		
		for (int i=0; i < data.numInstances(); i++) {
			double iragarpena = nbM.classifyInstance(data.instance(i));
			String labelName = data.classAttribute().value((int) iragarpena);
            String line = ((i + 1) + "  |  " + labelName);
            System.out.println(line);
            pw.println(line);
            
		}
	
	}
}

