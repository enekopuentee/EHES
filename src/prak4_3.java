
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import java.util.Random;

public class prak4_3 {

    public static void main(String[] args) throws Exception {
        // 1. DATUAK KARGATU
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // 2. KLASE MINORITARIOA IDENTIFIKATU
        int classIndex = data.classIndex();
        int minorityClassIdx = 0;
        double minCount = Double.MAX_VALUE;
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            int count = data.attributeStats(classIndex).nominalCounts[i];
            if (count < minCount && count > 0) {
                minCount = count;
                minorityClassIdx = i;
            }
        }

        // 3. STRATIFIED HOLD-OUT (%80 train / %20 test)
        Resample rs = new Resample();
        rs.setNoReplacement(true);
        rs.setSampleSizePercent(80);
        rs.setRandomSeed(42);
        rs.setInputFormat(data);
        Instances train = Filter.useFilter(data, rs);
        
        rs.setSampleSizePercent(20);
        rs.setInputFormat(data);
        Instances test = Filter.useFilter(data, rs);

        // 4. PARAMETRO EKORKETA (maxDepth eta numFeatures)
        int bestMaxDepth = -1;
        int bestNumFeatures = -1;
        double bestFScore = -1;

        int[] depthRange = {0, 5, 10, 15}; // 0 = mugagabea
        int[] featureRange = {1, 2, 3, 4}; // Atributu kopuruaren arabera

        System.out.println("Ekorketa hasten...");

        for (int depth : depthRange) {
            for (int features : featureRange) {
                RandomForest rf = new RandomForest();
                rf.setMaxDepth(depth);
                rf.setNumFeatures(features);

                rf.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                eval.evaluateModel(rf, test);

                // KLASE MINORITARIOAREN F-SCORE-A
                double fScore = eval.fMeasure(minorityClassIdx);

                if (fScore > bestFScore) {
                    bestFScore = fScore;
                    bestMaxDepth = depth;
                    bestNumFeatures = features;
                }
            }
        }

        // 5. EMAITZA OPTIMOAK INPRIMATU
        System.out.println("--- PARAMETRO OPTIMOAK ---");
        System.out.println("MaxDepth: " + bestMaxDepth);
        System.out.println("NumFeatures: " + bestNumFeatures);
        System.out.println("Minority Class F-Score: " + bestFScore);

        // 6. EREDU OPTIMOA GORDE
        RandomForest bestRF = new RandomForest();
        bestRF.setMaxDepth(bestMaxDepth);
        bestRF.setNumFeatures(bestNumFeatures);
        bestRF.buildClassifier(train);
        SerializationHelper.write("rf_optimoa.model", bestRF);

        // 7. EREDUA KARGATU ETA IRAGARPENAK EGIN (3. Praktikako logika)
        RandomForest loadedRF = (RandomForest) SerializationHelper.read("rf_optimoa.model");
        System.out.println("\n--- TEST MULTZOKO IRAGARPENAK ---");
        for (int i = 0; i < test.numInstances(); i++) {
            double pred = loadedRF.classifyInstance(test.instance(i));
            System.out.println("Instantzia " + i + " | Benetakoa: " + test.classAttribute().value((int)test.instance(i).classValue()) + 
                               " | Iragarpena: " + test.classAttribute().value((int)pred));
        }
    }
}
