using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations.Schema; //per fixare tipo Column (rompe ordinal come prima)
using Microsoft.ML; //per usare questa libreria scaricare il pacchetto Microsoft ML
//tasto destro sulla solution, manage NuGet packages..., Browse, Microsoft.ML
using Microsoft.ML.Data;
namespace _20200114_MediaCelluleTrimestrePrecedente
{
    class FeedBackTrainingData
    {
        [Column(ordinal: "0", name: "label")] //sistemare tipo Column
        public bool IsGood { get; set; }
        [Column(ordinal: "1", name: "label")]
        public string FeedBackText { get; set; }
    }
    class Program
    {
        static List<FeedBackTrainingData> trainingdata=
            new List<FeedBackTrainingData>();
        static List<FeedBackTrainingData> testData =
            new List<FeedBackTrainingData>();
            static void LoadTestData() //dati di test
        {
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is good", //valore oltre il limite massimo
                IsGood = true
            });
            testData.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = false
            });
        }
        static void LoadTrainingData() //dati di training
        {
            //sostituire poi con importazione file Excel
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "shitty", 
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Average ok", 
                IsGood = true
            });
        }
        static void Main(string[] args)
        {
            //1. carica i dati di training
            LoadTrainingData();

            //2. crea un oggetto di MLContext
            var mlContext = new MLContext();

            //3. converte i dato in IData View
            //LoadFromEnumerable era ReadFromEnumerable
            IDataView dataView = mlContext.Data.LoadFromEnumerable(trainingdata); //<FeedBackTrainingData> fra "LoadFromEnumerable" e "(trainingdata);" 
            
            //4. crea la pipeline

            var pipeline = mlContext.Transforms.
                          Text.FeaturizeText("Feedback", "Features")
                          .Append(mlContext.BinaryClassification.Trainers.FastTree
                          (numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));

            //5. train
            var model = pipeline.Fit(dataView);

            //6. testare con dati appositi, diversi da quelli di training
            LoadTestData();
            IDataView dataView1 = mlContext.Data.LoadFromEnumerable(trainingdata);
            var predictions = model.Transform(dataView1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine(metrics.Accuracy);
            Console.Read();

            //7. utilizzare il modello
            Console.WriteLine("Enter a feedback string: ");
            string feedbackstring = Console.Read().ToString();
            /*var predictionFunction = model.MakePredictionFunction
                                            <FeedBackTrainingData, FeedBackPrediction>
                                            (mlContext.Context);*/
            var feedbackinput = new FeedBackTrainingData();
            feedbackinput.FeedBackText = feedbackstring;
            predictionFunction.predict(feedbackinput);
        }
    }
}