import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

public class SHO {

		public static void main (String [] args) throws Exception {
			ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
			Instances data = source.getDataSet();
			if(data.classIndex()== -1) {
				data.setClassIndex(data.numAttributes()-1);
			}
			
			StratifiedRemoveFolds srf = new StratifiedRemoveFolds();
			srf.setNumFolds(5);
			srf.setFold(1);
			srf.setInvertSelection(true);
			srf.setInputFormat(data);
			Instances trainData = Filter.useFilter(data, srf);
			
			srf = new StratifiedRemoveFolds();
			srf.setNumFolds(5);
			srf.setFold(1);
			srf.setInvertSelection(false);
			srf.setInputFormat(data);
			Instances devData = Filter.useFilter(data, srf);
			
			
		}
		private static void saver(Instances data, String args) throws Exception{
			ArffSaver sv = new ArffSaver(); 
			sv.setInstances(data); 
			sv.setFile(new File(args)); 
			sv.writeBatch();
		}
}
