
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.instance.Randomize;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;
public class HoldOut {

public static void main (String[] args) throws Exception{

	DataSource source = new DataSource(args[0]);
	Instances data = source.getDataSet();
	if (data.classIndex() == -1)
		data.setClassIndex(data.numAttributes() - 1);

	//RANDOMIZE

	Randomize r = new Randomize();
	r.setRandomSeed(1);
	r.setInputFormat(data);
	Instances randomData = Filter.useFilter(data, r);

	//TRAIN

	RemovePercentage rp = new RemovePercentage();
	rp.setPercentage(34.0);
	rp.setInvertSelection(true);
	rp.setInputFormat(randomData);
	Instances trainData = Filter.useFilter(randomData, rp);
	
	//TEST

	rp.setInvertSelection(false);
	rp.setInputFormat(randomData);
	Instances testData = Filter.useFilter(randomData, rp);

	//SAILKATZAILEA
	NaiveBayes nb = new NaiveBayes();
	nb.buildClassifier(trainData);
	Evaluation eval = new Evaluation(testData);
	eval.evaluateModel(nb, testData);

	try(FileWriter wr = new FileWriter(args[1], true)) {
		wr.write("Exekuzio Data");
	}catch (IOException e) {
		e.printStackTrace();

	}
}
}
