import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ModelGenerator {

    int batchSize;
    int numEpochs;
    public ModelGenerator(int batchSize, int numEpochs){
        this.numEpochs=numEpochs;
        this.batchSize=batchSize;
    }

    public void fitModel(DataSetIterator iterator,MultiLayerNetwork net){
        for(int i=0;i<numEpochs;i++) {
            System.out.println("Model training for epoch " + i);
            iterator.forEachRemaining(net::fit);
            //System.out.println(net.evaluate(iterator).accuracy());
            iterator.reset();
        }
    }

}
