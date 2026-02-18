import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.core.EuclideanDistance;
import weka.core.ManhattanDistance;
import weka.core.SelectedTag;
import weka.core.DistanceFunction;

public class prak4_1 {
	
	public static void main(String[] args) throws Exception {
		ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		
		Resample rs = new Resample();
		rs.setNoReplacement(true);
		rs.setSampleSizePercent(66);
		rs.setRandomSeed(1);
		rs.setInputFormat(data);
		Instances train = Filter.useFilter(data, rs);
		
		rs.setSampleSizePercent(34);
		rs.setInputFormat(data);
		Instances test = Filter.useFilter(data, rs);
		
		
	
		int[] kValues = {1, 3, 5, 7, 9}; 
        int[] wValues = {IBk.WEIGHT_NONE, IBk.WEIGHT_INVERSE}; 
        DistanceFunction[] dValues = {new EuclideanDistance(), new ManhattanDistance()}; 

    	double bestFScore=-1;
		int bestK=-1;
		int bestW=-1;
		String bestD="";
     


      
        for (int k : kValues) {
            for (int w : wValues) {
                for (DistanceFunction d : dValues) {
                    
                    IBk knn = new IBk();
                    knn.setKNN(k); // 
                    knn.setDistanceWeighting(new SelectedTag(w, IBk.TAGS_WEIGHTING)); 
                    knn.getNearestNeighbourSearchAlgorithm().setDistanceFunction(d); 

                   
                    knn.buildClassifier(train);
                    Evaluation eval = new Evaluation(train);
                    eval.evaluateModel(knn, test);

                    
                    double currentFScore = eval.weightedFMeasure();
                    
                    String distName = (d instanceof EuclideanDistance) ? "Euclidean" : "Manhattan";
                    //String weightName = (w == IBk.WEIGHT_NONE) ? "None   " : "Inverse";
                    

                  
                    if (currentFScore > bestFScore) {
                        bestFScore = currentFScore;
                        bestK = k;
                        bestW = w;
                        bestD = distName;
                    }
                }
            }
        }

        
        System.out.println("---------------------------------------------------------");
        System.out.println("Mejor k: " + bestK);
        System.out.println("Mejor Distancia (d): " + bestD);
        System.out.println("Mejor Peso (w): " + (bestW == IBk.WEIGHT_NONE ? "None" : "Inverse"));
        System.out.println("Mejor F-Score: " + String.format("%.4f", bestFScore));
		
	}

}
