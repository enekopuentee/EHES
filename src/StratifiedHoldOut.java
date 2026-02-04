import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.File;

public class StratifiedHoldOut {
    public static void main(String[] args) {

        try {
            // Datuak kargatu
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // StratifiedRemoveFolds konfiguratu (5 fold sortuko ditugu, %20 fold bakoitzeko)

            //Dev
            StratifiedRemoveFolds filterDev = new StratifiedRemoveFolds();
            filterDev.setNumFolds(5);
            filterDev.setFold(1);
            filterDev.setInvertSelection(false); // Fold bakarra hautatu
            filterDev.setInputFormat(data);
            Instances devData = Filter.useFilter(data, filterDev);

            // Train
            StratifiedRemoveFolds filterTrain = new StratifiedRemoveFolds();
            filterTrain.setNumFolds(5);
            filterTrain.setFold(1);
            filterTrain.setInvertSelection(true); // Gainontzeko guztiak hautatu
            filterTrain.setInputFormat(data);
            Instances trainData = Filter.useFilter(data, filterTrain);

            // Fitxategiak gorde
            saveArff(trainData, args[1]);
            saveArff(devData, args[2]);

            System.out.println("Partiketa ondo burutu da:");
            System.out.println("Train (%80): " + args[1]);
            System.out.println("Dev (%20): " + args[2]);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void saveArff(Instances data, String path) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(path));
        saver.writeBatch();
    }
}
