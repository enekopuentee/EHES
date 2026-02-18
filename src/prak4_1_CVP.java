import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Arrays;

public class prak4_1_CVP {

    public static void main(String[] args) {
        try {
            // 1. Datuak kargatu
            DataSource source = new DataSource("balance-scale.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // 2. Meta-sailkatzailea sortu
            CVParameterSelection cvp = new CVParameterSelection();
            
            // Oinarrizko sailkatzailea ezarri (IBk)
            IBk knn = new IBk();
            cvp.setClassifier(knn);

            // 3. Parametroen tarteak definitu (Ekorketa konfiguratu)
            // Sintaxia: "parametroarenIzena hasiera bukaera urratsa"
            // 'K' -> KNN (Auzokideak)
            cvp.addCVParameter("K 1 10 10"); 
            
            // 4. Ebaluazio ezaugarriak konfiguratu
            cvp.setNumFolds(10); // 10-fold cross validation 
            
            // 5. Optimizazioa exekutatu (Honek konbinazio onena bilatzen du)
            cvp.buildClassifier(data);

            // 6. Emaitzak eta parametro onenak lortu
            String[] bestParams = cvp.getBestClassifierOptions();
            
            System.out.println("--- CVParameterSelection Emaitzak ---");
            System.out.println("Parametro optimoak: " + Arrays.toString(bestParams));
            System.out.println("Ereduaren laburpena:\n" + cvp.toString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    /**
     * 2. ARIKETA: Parametro ekorketa meta-sailkatzaileak erabiliz (Teoria)
     * * CVParameterSelection eta GridSearch liburutegien arteko konparaketa:
     * * 1. HELBURU-FUNTZIOA:
     * - CVParameterSelection: Cross-validation bidezko akatsa minimizatzea edo 
     * asmatze-tasa (Accuracy) maximizatzea du helburu nagusi.
     * - GridSearch: Definitutako metrika espezifiko bat (F-measure, Accuracy, 
     * AUC, etab.) maximizatzea edo minimizatzea ahalbidetzen du.
     * * 2. SARRERAK (Zer sartzen da?):
     * - CVParameterSelection: Oinarrizko sailkatzailea eta parametro-tarteak 
     * jasotzen ditu (Adibidez: "K 1 10 10" -> k-NN-ren K parametroa 1etik 10era).
     * - GridSearch: Oinarrizko sailkatzailea eta bi parametro-ardatz (X eta Y) 
     * definitzen dira, bakoitzaren Min, Max eta Step (urratsa) balioekin.
     * * 3. EMAITZA (Zer lortzen dugu?):
     * - CVParameterSelection: Parametro konbinazio onena eta datu multzo osoarekin 
     * entrenatutako modelo optimoa bueltatzen du.
     * - GridSearch: Puntuazio onena eman duen "sareko" (grid) puntua identifikatzen du 
     * eta horri dagokion modelo optimizatua itzultzen du.
     * * 4. IDENTIKOAK AL DIRA?:
     * Kontzeptualki BAI, biek bilaketa exhaustiboa (brute force) egiten baitute 
     * parametroen artean. Baina praktikan EZ:
     * - CVParameterSelection malguagoa da parametro asko (2 baino gehiago) aldi 
     * berean probatzeko, baina ebaluazio metrika gutxiago ditu.
     * - GridSearch indartsuagoa da bi parametro zehatzen arteko erlazioa aztertzeko 
     * eta metrika askoz gehiago onartzen ditu (F-measure barne).
     */

}