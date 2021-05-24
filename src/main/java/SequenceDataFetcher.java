import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;

//class for creating a dataset iterator
public class SequenceDataFetcher extends BaseDataFetcher {

    private DataSet data;

    public SequenceDataFetcher(DataSet data){
        this.data = data;
        this.totalExamples = data.numExamples();
        this.numOutcomes = data.numOutcomes();
        this.inputColumns = data.numInputs();
    }

    @Override
    public void fetch(int numExamples) {
        int from = this.cursor;
        int to = this.cursor + numExamples;
        if (to > this.totalExamples) {
            to = this.totalExamples;
        }
        //System.out.println(data.asList().subList(from,to));
        this.initializeCurrFromList(data.asList().subList(from,to));
        this.cursor += numExamples;
    }
}
