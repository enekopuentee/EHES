import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Date;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
// ATRIBUTU HAUTAPENARAKO IMPORT-AK
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;

public class HoldOut {

    public static void main(String[] args) {

        try {
            // Datuak kargatu
            DataSource source = new DataSource(args[0]);
            Instances data = source.getDataSet();

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // DATUEN BANAKETA (Hold-out %66)
            Randomize randomize = new Randomize();
            randomize.setInputFormat(data);
            Instances randomizedData = Filter.useFilter(data, randomize);


            RemovePercentage removeTrain = new RemovePercentage();
            removeTrain.setPercentage(34.0);
            removeTrain.setInputFormat(randomizedData);
            Instances trainData = Filter.useFilter(randomizedData, removeTrain);


            RemovePercentage removeTest = new RemovePercentage();
            removeTest.setPercentage(34.0);
            removeTest.setInvertSelection(true);
            removeTest.setInputFormat(randomizedData);
            Instances testData = Filter.useFilter(randomizedData, removeTest);

            // ATRIBUTU HAUTAPENA (Attribute Selection)
            System.out.println("Atributu hautapena aplikatzen...");
            AttributeSelection filter = new AttributeSelection();
            filter.setEvaluator(new CfsSubsetEval());
            filter.setSearch(new BestFirst());
            filter.setInputFormat(trainData);

            // Bi multzoak iragazi
            Instances trainFiltered = Filter.useFilter(trainData, filter);
            Instances testFiltered = Filter.useFilter(testData, filter);

            // Eredua entrenatu eta ebaluatu (Iragazitako datuekin)
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(trainFiltered);

            Evaluation eval = new Evaluation(trainFiltered);
            eval.evaluateModel(nb, testFiltered);

            // Klase minoritarioa identifikatu
            int minoritaryClassIndex = getMinoritaryClass(trainFiltered);

            // Emaitzak prestatu
            StringBuilder sb = new StringBuilder();
            sb.append("======================================================\n");
            sb.append("HOLD-OUT (%66 Train / %34 Test) + ATTRIBUTE SELECTION\n");
            sb.append("======================================================\n");
            sb.append("Exekuzio data: ").append(new Date().toString()).append("\n");
            sb.append("Argumentuak: ").append(args[0]).append(", ").append(args[1]).append("\n");
            sb.append("Atributu kopurua: ").append(data.numAttributes()).append(" -> ").append(trainFiltered.numAttributes()).append("\n");
            sb.append("------------------------------------------------------\n\n");

            sb.append("NAHASMEN MATRIZEA:\n");
            sb.append(eval.toMatrixString()).append("\n");

            sb.append("KLASE MINORITARIOAREN METRIKAK (")
                    .append(trainFiltered.classAttribute().value(minoritaryClassIndex)).append("):\n");
            sb.append(String.format("  Precision: %.4f%n", eval.precision(minoritaryClassIndex)));
            sb.append(String.format("  Recall:    %.4f%n", eval.recall(minoritaryClassIndex)));
            sb.append(String.format("  F-Measure: %.4f%n", eval.fMeasure(minoritaryClassIndex)));

            sb.append("\nWEIGHTED AVG METRIKAK:\n");
            sb.append(String.format("  Weighted Avg Precision: %.4f%n", eval.weightedPrecision()));
            sb.append(String.format("  Weighted Avg Recall:    %.4f%n", eval.weightedRecall()));
            sb.append(String.format("  Weighted Avg F-Measure: %.4f%n", eval.weightedFMeasure()));

            // output
            System.out.println(sb.toString());
            PrintWriter pw = new PrintWriter(new FileWriter(args[1]));
            pw.print(sb.toString());
            pw.close();

            System.out.println("\n>>> Prozesua ondo burutu da.");

        } catch (Exception e) {
            System.err.println("ERROREA: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static int getMinoritaryClass(Instances data) {
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        int minIndex = 0;
        int minCount = counts[0];
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] < minCount) {
                minCount = counts[i];
                minIndex = i;
            }
        }
        return minIndex;
    }
}