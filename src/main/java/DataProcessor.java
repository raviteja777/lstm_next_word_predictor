import org.datavec.api.transform.transform.integer.IntegerToOneHotTransform;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.deeplearning4j.text.stopwords.StopWords;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;


public class DataProcessor implements Serializable {

    transient private final File file;
    transient private final DataSetIterator dataSetIterator;
    private final List<String> corpus;
    List<String> stopWords = StopWords.getStopWords();
    transient private KerasTokenizer tokenizer;
    private final int seqLen;

    public DataProcessor(String filepath,int batch) throws IOException {
        this.file = new File(filepath);
        if(!file.exists()){
            throw new FileNotFoundException("Input file not found");
        }
        corpus = createCorpus();
        tokenizer = initTokenizer();
        DataSet dataSet = prepareDataSet();
        //System.out.println(dataSet.asList().size());
        dataSetIterator = createIterator(batch,dataSet);
        seqLen = dataSetIterator.inputColumns();
    }

    public List<String> createCorpus() throws IOException {
        return Files.lines(file.toPath())
                .filter(l -> !l.isBlank())
                .map(l -> l.trim().toLowerCase().replaceAll("[^\\w\\s]", "").split("\\s+"))
                .flatMap(arr -> Arrays.stream(arr).filter(x -> !stopWords.contains(x)))
                .distinct().sorted().collect(Collectors.toList());
    }

    //set keras tokenizer
    public KerasTokenizer initTokenizer(){
        String[] texts = corpus.toArray(new String[numWords]);
        tokenizer = new KerasTokenizer();
        tokenizer.fitOnTexts(texts);
        return tokenizer;
    }

    public DataSetIterator createIterator(int batch,DataSet data){
        BaseDataFetcher fetcher = new SequenceDataFetcher(data);
        return new BaseDatasetIterator(batch,data.numExamples(),fetcher);
        //refer IrisDataFetcher
    }

    public DataSet prepareDataSet() throws IOException {
        Integer[][] sequences = tokenizer.textsToSequences(processData().toArray(new String[0]));
        //Arrays.stream(sequences).forEach(x->System.out.println(Arrays.deepToString(x)));
        int maxLen =Arrays.stream(sequences).max(Comparator.comparingInt(a -> a.length)).orElseThrow().length;
        return DataSet.merge(Arrays.stream(sequences).map(x->padSequencesToDataset(x,maxLen))
                .collect(Collectors.toList()));
    }


    public List<String> processData() throws IOException {

        return Files.lines(file.toPath())
                .filter(l -> !l.isBlank())
                .map(l -> l.trim().toLowerCase().replaceAll("[^\\w\\s]", "").split("\\s+"))
                .map(arr -> Arrays.stream(arr).filter(x -> !stopWords.contains(x)).collect(Collectors.toList()))
                .filter(x->x.size()>2)
                .flatMap(arr->spanArrays(arr.toArray(new String[0]),3).stream())
                .collect(Collectors.toList());
    }

    //helper method to create sub sequences
    //eg : his name is ram --> [ [his,name] ,[his,name,is] ,[his,name ,is,ram] ]
    private List<String> spanArrays(String[] arr,int min) {
        List<String> arrList = new ArrayList<>();
        for(int i=min;i<arr.length;i++){
            arrList.add(Arrays.stream(arr,0,i).collect(Collectors.joining(" ")));
        }
        return arrList;
    }

    //pad 0 s to make sequences of uniform length
    //create a dataset as {features(sequences) , labels (1 hot encoded vector for the final word in each sequence)}
    private DataSet padSequencesToDataset(Integer[] seq,int maxLen){
        Integer[] seq1 = Arrays.copyOfRange(seq,0,maxLen-1);
        Arrays.fill(seq1,seq.length-1,maxLen-1,0);
        int res = seq[seq.length-1];
        INDArray features = Nd4j.create(List.of(seq1));
        features = features.reshape(1,features.size(0));

        IntegerToOneHotTransform intoh = new IntegerToOneHotTransform("res",0,corpus.size()-1);
        INDArray labels = Nd4j.create((List<Integer>)intoh.map(res));
        labels = labels.reshape(1,labels.size(0));

        return new DataSet(features,labels);
    }

    public DataSetIterator getDataSetIterator(){
        return dataSetIterator;
    }

    public List<String> getCorpus(){
        return corpus;
    }


    //process a single test line
    //convert to INDArray for prediction
    public INDArray processTestString(String data){
        if(data.isBlank()||data.trim().split("\\s+").length<2){
            throw new IllegalArgumentException("String should contain atleast 2 words");
        }
        data = data.trim().toLowerCase().replaceAll("[^\\w\\s]", "");

        //convert text to sequences using tokenizer
        Integer[][] sequences = tokenizer.textsToSequences(new String[]{data});
        //pad 0s to sequences to make it std length
        //int seqLen = dataSetIterator.inputColumns();
        Integer[] seq = Arrays.copyOfRange(sequences[0],0,seqLen);
        Arrays.fill(seq,sequences[0].length,seqLen,0);
        int[] batch = new int[]{1};
        //convert to ND Array
        INDArray input = Nd4j.create(Arrays.asList(seq));
        input = input.reshape(1,seq.length);
        return input;
    }

}
