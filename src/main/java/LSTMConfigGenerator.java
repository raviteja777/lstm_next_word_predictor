import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LSTMConfigGenerator {

    private final int numInputs;          //input feature length
    private final int numLabels ;       //output length

    public LSTMConfigGenerator(int inputSize, int outputSize){
        numInputs = inputSize; //total number of words in vocabulary -- for embedding layer
        numLabels = outputSize;
    }

    //method to configure and build network
    public MultiLayerNetwork buildNetwork(){

        //Number of units in each LSTM layer
        int lstmLayerSize = numInputs*2;
        //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int tbpttLength = 50;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new EmbeddingSequenceLayer.Builder()
                        .nIn(numInputs+1)
                        .nOut(lstmLayerSize)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new LSTM.Builder()
                        .nIn(lstmLayerSize)
                        .nOut((int) (lstmLayerSize *0.75))
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .nIn((int) (lstmLayerSize *0.75))
                        .nOut(numLabels)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTBackwardLength(tbpttLength)
                .tBPTTForwardLength(tbpttLength)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));
        System.out.println(net.summary());
        return net;
    }


}
