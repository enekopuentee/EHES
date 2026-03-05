
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

        // 2. Meta-sailkatzailea sortu (CVParameterSelection)
        CVParameterSelection cvp = new CVParameterSelection();
        cvp.setClassifier(new RandomForest()); // RandomForest oinarri gisa
        cvp.setNumFolds(5);                    // 5-fold cross validation 

        // 3. PARAMETROAK DEFINITU
        // maxDepth (sakonera): 0-tik (mugagabea) 20-ra probatu
        cvp.addCVParameter("maxDepth 0 20 5"); 
        // numFeatures (atributuak): 1-etik 5-era probatu
        cvp.addCVParameter("numFeatures 1 5 5"); 

        // 4. EKORKETA EXEKUTATU
        System.out.println("Ekorketa prozesua hasten (honek luzeago jo dezake)...");
        cvp.buildClassifier(data);

        // 5. EMAITZA OPTIMOAK INPRIMATU
        String bestOptions = Utils.joinOptions(cvp.getBestClassifierOptions());
        System.out.println("-------------------------------------------------");
        System.out.println("Parametro optimoak aurkituta: " + bestOptions);
        System.out.println("-------------------------------------------------");

        // 6. EREDU IRAGARLE OPTIMOA GORDE
        SerializationHelper.write("RF_optimoa.model", cvp);
        System.out.println("Eredua 'RF_optimoa.model' fitxategian gorde da.");

        // 7. EREDUA KARGATU ETA IRAGARPENAK EGIN (3. Praktikako logika)
        CVParameterSelection loadedCvp = (CVParameterSelection) SerializationHelper.read("RF_optimoa.model");
        
        System.out.println("\nIragarpenak (lehenengo 5 instantziak):");
        for (int i = 0; i < 5 && i < data.numInstances(); i++) {
            double pred = loadedCvp.classifyInstance(data.instance(i));
            String real = data.classAttribute().value((int) data.instance(i).classValue());
            String predicted = data.classAttribute().value((int) pred);
            System.out.println("Instantzia " + i + " | Benetakoa: " + real + " | Iragarpena: " + predicted);
        }
    }
}
