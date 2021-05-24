import org.bytedeco.javacpp.tools.Slf4jLogger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Scanner;

public class LSTMPredictorMain {

    private final static int batchSize = 32;
    private final static int epochs = 5;

    private final static String fp = System.getenv("PWD")+"/data/sample1.txt";
    private final static String saveModelPath = System.getenv("PWD")+"/saved_models";

    private final static Slf4jLogger logger = new Slf4jLogger(LSTMPredictorMain.class);

    public static void main(String[] args) throws IOException {
        logger.info("Reading data from file--- "+fp);
        logger.info("======Processing Data======");
        DataProcessor proc = new DataProcessor(fp,batchSize);
        List<String> corpus = proc.getCorpus();
        DataSetIterator dataSetIterator = proc.getDataSetIterator();

        //trained model
        MultiLayerNetwork network = createModel(corpus,dataSetIterator);
        //save model files
        saveModelFiles(network,proc);
        //test a phrase
        testPhrase(proc,corpus,network);
    }



    public static MultiLayerNetwork createModel(List<String> corpus,DataSetIterator iter){
        //configure neural network
        logger.info("Configuring network");
        LSTMConfigGenerator lstm = new LSTMConfigGenerator(corpus.size(), iter.totalOutcomes());
        MultiLayerNetwork network = lstm.buildNetwork();

        //train model
        logger.info("Training model ......");
        ModelGenerator modelGenerator = new ModelGenerator(batchSize,epochs);
        modelGenerator.fitModel(iter,network);
        return network;
    }

    public static void saveModelFiles(MultiLayerNetwork network,DataProcessor proc) throws IOException {
        logger.info("====Saving Model files======");

        String timeStamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_hhmmss"));
        String filepath = saveModelPath+"/model_files_"+timeStamp+"/";
        Files.createDirectory(Path.of(filepath));
        //save model file
        String modelFileName = "lstm_"+timeStamp+".model";
        network.save(new File(filepath+modelFileName));
        logger.info("Model saved at path :"+filepath+modelFileName);
        //save processor
        String procFileName = "processor_"+timeStamp+".ser";
        ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(filepath+procFileName));
        oos.writeObject(proc);
        oos.close();
        logger.info("Processor saved at path :"+filepath+procFileName);
    }

    //test a phrase
    public static void testPhrase(DataProcessor proc,List<String> corpus,MultiLayerNetwork network){
        logger.info("===Predicting test phrase =======");
        logger.info("===== Enter the test phrase ======");
        String testStr = new Scanner(System.in).nextLine().trim();
        INDArray processedData = proc.processTestString(testStr);
        System.out.println(processedData);
        INDArray output = network.rnnTimeStep(processedData);
        int ind = output.argMax(1).toIntVector()[0];
        logger.info("test inp: "+testStr);
        logger.info("Max ouput at index: "+ind);
        logger.info("Max output score: "+output.getDouble(ind));
        logger.info("Predicted next word: "+corpus.get(ind));
    }


}


