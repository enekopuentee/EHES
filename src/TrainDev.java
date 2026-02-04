import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;

public class TrainDev {
    public static void main(String[] args) {

        try {
            // Train eta Dev kargatu
            DataSource sourceTrain = new DataSource(args[0]);
            Instances train = sourceTrain.getDataSet();
            train.setClassIndex(train.numAttributes() - 1);

            DataSource sourceDev = new DataSource(args[1]);
            Instances dev = sourceDev.getDataSet();
            dev.setClassIndex(dev.numAttributes() - 1);

            // Naive Bayes entrenatu
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train);

            // Ebaluazioa egin Dev multzoarekin
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(nb, dev);

            // Emaitzak fitxategian gorde
            PrintWriter pw = new PrintWriter(new FileWriter(args[2]));
            pw.println("=== TRAIN VS DEV EBALUAZIOA ===");
            pw.println("Data: " + new Date().toString());
            pw.println("Train path: " + args[0]);
            pw.println("Dev path: " + args[1]);
            pw.println("--------------------------------");
            pw.println("\nNAHASMEN MATRIZEA:");
            pw.println(eval.toMatrixString());
            pw.printf("ACCURACY: %.4f %%%n", eval.pctCorrect());
            pw.close();

            // Terminalean ere erakutsi
            System.out.println("Ebaluazioa amaituta. Emaitzak hemen: " + args[2]);
            System.out.println(eval.toMatrixString());
            System.out.println("Accuracy: " + eval.pctCorrect() + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
