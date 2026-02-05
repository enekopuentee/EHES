import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class prak3ariketa1 {
	public static void main(String[] args) throws Exception {
		
			DataSource source = new DataSource(args[0]);
			Instances data = source.getDataSet();
			
			if(data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			
			NaiveBayes nb = new NaiveBayes();
			nb.buildClassifier(data);
			
			SerializationHelper.write(args[1],nb);
			System.out.println("Eredua gorde da:" +args[1]);
			
			FileWriter fw = new FileWriter(args[2]);
			PrintWriter pw = new PrintWriter(fw);
			
			//10-FCV
			
			Evaluation evalFCV = new Evaluation(data);
			evalFCV.crossValidateModel(nb, data, 10, new Random(1));
			System.out.println("FCV emaitza");
			pw.println(evalFCV.toSummaryString());
			pw.println(evalFCV.toMatrixString());
			System.out.println(evalFCV.toSummaryString());
			System.out.println(evalFCV.toMatrixString());
			
			//Hold-Out %70
			
			Randomize rm = new Randomize();
			rm.setRandomSeed(42);
			rm.setInputFormat(data);
			Instances randData = Filter.useFilter(data, rm);
			
			RemovePercentage rp = new RemovePercentage();
			rp.setPercentage(30.0);
			rp.setInvertSelection(false);
			rp.setInputFormat(randData);
			Instances trainData = Filter.useFilter(randData, rp);
			
			rp.setPercentage(30.0);
			rp.setInvertSelection(true);
			rp.setInputFormat(randData);
			Instances testData = Filter.useFilter(randData, rp);
			
			NaiveBayes nbHO = new NaiveBayes();
			nbHO.buildClassifier(trainData);
			
			Evaluation evalHO = new Evaluation(trainData);
			evalHO.evaluateModel(nbHO, testData);
			
			System.out.println("Hold Out emaitza");
			System.out.println(evalHO.toSummaryString());
			System.out.println(evalHO.toMatrixString());
			pw.println(evalHO.toSummaryString());
			pw.println(evalHO.toMatrixString());
			
			
			
	}
}
