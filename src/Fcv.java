import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;

public class Fcv {

    public static void main(String[] args) {

        try {
            // Datuak kargatu
            System.out.println("Datuak kargatzen: " + args[0]);
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // ATRIBUTU HAUTAPENA (Attribute Selection)
            System.out.println("Atributu hautapena aplikatzen...");
            AttributeSelection filter = new AttributeSelection();
            filter.setEvaluator(new CfsSubsetEval());
            filter.setSearch(new BestFirst());
            filter.setInputFormat(data);


            Instances filteredData = Filter.useFilter(data, filter);
            System.out.println("Atributu kopurua: " + data.numAttributes() + " -> " + filteredData.numAttributes());

            // Naive Bayes + 5-fold Cross Validation

            NaiveBayes nb = new NaiveBayes();
            Evaluation evaluator = new Evaluation(filteredData);
            evaluator.crossValidateModel(nb, filteredData, 5, new Random(1));

            // Emaitzak
            StringBuilder sb = new StringBuilder();
            sb.append("======================================================\n");
            sb.append("=== Evaluation Results (Attribute Selection + 5-fCV) ===\n");
            sb.append("======================================================\n");
            sb.append("Date: ").append(new Date().toString()).append("\n");
            sb.append("Arguments: ").append(args[0]).append(", ").append(args[1]).append("\n");
            sb.append("Original attributes: ").append(data.numAttributes()).append("\n");
            sb.append("Selected attributes: ").append(filteredData.numAttributes()).append("\n\n");

            // nahasketa matrizea
            sb.append("NB 5-fCV nahasmen matrizea:\n");
            sb.append(evaluator.toMatrixString()).append("\n");


            sb.append(String.format("Correctly Classified Instances   %.4f %%\n", evaluator.pctCorrect()));
            sb.append(String.format("Incorrectly Classified Instances %.4f %%\n", evaluator.pctIncorrect()));
            sb.append(String.format("Kappa statistic                  %.4f\n", evaluator.kappa()));
            sb.append(String.format("Mean absolute error              %.4f\n", evaluator.meanAbsoluteError()));
            sb.append(String.format("Root mean squared error          %.4f\n", evaluator.rootMeanSquaredError()));
            sb.append(String.format("Relative absolute error          %.4f %%\n", evaluator.relativeAbsoluteError()));
            sb.append(String.format("Root relative squared error      %.4f %%\n", evaluator.rootRelativeSquaredError()));


            sb.append("\n=== Detailed Accuracy By Class (Precision) ===\n");
            for (int i = 0; i < filteredData.numClasses(); i++) {
                sb.append(String.format("Class %-15s Precision: %.4f\n",
                        filteredData.classAttribute().value(i), evaluator.precision(i)));
            }
            sb.append(String.format("Weighted Avg. Precision:        %.4f\n", evaluator.weightedPrecision()));

            // 5. output
            String resultadoFinal = sb.toString();


            System.out.println(resultadoFinal);


            PrintWriter pw = new PrintWriter(new FileWriter(args[1]));
            pw.print(resultadoFinal);
            pw.close();

            System.out.println("\n>>> Prozesua ondo burutu da. Emaitzak fitxategian gordeta: " + args[1]);

        } catch (Exception e) {
            System.err.println("ERROREA: " + e.getMessage());
            e.printStackTrace();
        }
    }
}