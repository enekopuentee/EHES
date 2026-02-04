import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Praktika1 {

    public static void main(String[] args) {

        try {
            // Datu-sorta kargatu (DataSource)
            String path = args[0];
            DataSource source = new DataSource(path);
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            System.out.println("=== DATU-SORTAREN KARAKTERIZAZIOA ===");

            //path-a
            System.out.println("1. Fitxategiaren bidea: " + path);

            //Instantzia kopurua
            System.out.println("2. Instantzia kopurua: " + data.numInstances());

            //Atributu kopurua
            System.out.println("3. Atributu kopurua: " + data.numAttributes());

            // Lehenengo atributuaren balio ezberdin kopurua
            AttributeStats firstAttrStats = data.attributeStats(0);
            System.out.println("4. Lehen atributuaren (0) balio ezberdin kopurua: " + firstAttrStats.distinctCount);

            //Azken atributuaren (klasea) analisia
            int classIndex = data.classIndex();
            Attribute classAttr = data.attribute(classIndex);
            AttributeStats classStats = data.attributeStats(classIndex);

            System.out.println("5. Azken atributuaren (klasea) balioak eta maiztasunak:");

            int minCount = Integer.MAX_VALUE;
            String minoritaryClass = "";

            for (int i = 0; i < classAttr.numValues(); i++) {
                String name = classAttr.value(i);
                int count = classStats.nominalCounts[i];
                System.out.println("   - " + name + ": " + count);

                // Klase minoritarioa bilatu
                if (count < minCount) {
                    minCount = count;
                    minoritaryClass = name;
                }
            }
            System.out.println("   -> Klase minoritarioa: " + minoritaryClass + " (" + minCount + " instantzia)");

            // Azken aurreko atributuaren missing value kopurua
            int penultIndex = data.numAttributes() - 2;
            AttributeStats penultStats = data.attributeStats(penultIndex);
            System.out.println("6. Azken aurreko atributuaren (" + penultIndex + ") missing value kopurua: " + penultStats.missingCount);

        } catch (Exception e) {
            System.err.println("Errorea gertatu da: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
