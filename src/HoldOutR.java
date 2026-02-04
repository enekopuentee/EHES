
import java.util.ArrayList;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class HoldOutR {

    public static void main(String[] args) throws Exception {
        // 1. Cargamos los datos una sola vez
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // Lista para guardar el Recall de la clase minoritaria de cada iteración
        ArrayList<Double> listaRecalls = new ArrayList<>();
        int numRepeticiones = 50;
        
        // El índice de la clase minoritaria en adult solía ser 1 (">50K")
        // Cámbialo si tu dataset usa otro índice
        int claseMinoritaria = 1; 
        

        System.out.println("Ejecutando Repeated Hold-Out (50 veces)...");

        for (int i = 0; i < numRepeticiones; i++) {
            // RANDOMIZE con semilla variable basada en la iteración i
            Randomize r = new Randomize();
            r.setRandomSeed(i); 
            r.setInputFormat(data);
            Instances randomData = Filter.useFilter(data, r);

            // TRAIN (66%)
            RemovePercentage rp = new RemovePercentage();
            rp.setPercentage(34.0);
            rp.setInvertSelection(true);
            rp.setInputFormat(randomData);
            Instances trainData = Filter.useFilter(randomData, rp);

            // TEST (34%)
            rp.setInvertSelection(false);
            rp.setInputFormat(randomData);
            Instances testData = Filter.useFilter(randomData, rp);

            // SAILKATZAILEA (Naive Bayes)
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(trainData);

            // EVALUATION
            Evaluation eval = new Evaluation(trainData); // Se recomienda inicializar con train
            eval.evaluateModel(nb, testData);

            // Obtener Recall de la clase minoritaria y guardarlo
            listaRecalls.add(eval.recall(claseMinoritaria));
        }

        // --- CÁLCULOS ESTADÍSTICOS ---
        
        // Media
        double suma = 0;
        for (double val : listaRecalls) suma += val;
        double media = suma / numRepeticiones;

        // Desviación Típica (Standard Deviation)
        double sumaVarianza = 0;
        for (double val : listaRecalls) {
            sumaVarianza += Math.pow(val - media, 2);
        }
        double stdev = Math.sqrt(sumaVarianza / numRepeticiones);

        // Resultados por consola
        System.out.println("\n--- RESULTADOS FINALES ---");
        System.out.println("Media Recall (Clase " + claseMinoritaria + "): " + String.format("%.4f", media));
        System.out.println("Desviación Típica: " + String.format("%.4f", stdev));
    }
}