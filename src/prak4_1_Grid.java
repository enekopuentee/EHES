//import weka.classifiers.meta.GridSearch;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;

public class prak4_1_Grid {

    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("balance-scale.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // 1. GridSearch objektua sortu
            //GridSearch gs = new GridSearch();
            
            // 2. Oinarrizko sailkatzailea konfiguratu (IBk)
            IBk knn = new IBk();
            //gs.setClassifier(knn);

            // 3. X Ardatza konfiguratu (Auzokideak - k)
            // "K" parametroa IBk klasifikadorearena da
            //gs.setXProperty("classifier.KNN");
            //gs.setXMin(1.0);
            //gs.setXMax(10.0);
            //gs.setXStep(1.0);
            //gs.setXExpression("I"); // Balioa bere horretan erabili (identitatea)

            // 4. Ebaluazioa konfiguratu (10-fold CV eta F-measure)
            //gs.setEvaluation(new SelectedTag(GridSearch.EVALUATION_FMEASURE, GridSearch.TAGS_EVALUATION));
            
            // 5. Optimizazioa exekutatu
            //gs.buildClassifier(data);

            // 6. Emaitza onenak ikusi
            System.out.println("Parametro optimoak aurkituta:");
            //System.out.println(gs.getBestClassifier().toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}