using Microsoft.Extensions.AI;
using OllamaSharp;
using System.Numerics.Tensors;

//create a chat client
IChatClient client =
    new OllamaApiClient(new Uri("http://localhost:11434"), "llama3.2:3b");
//create an embedding generatot (text-embedding-3-small is an example)
IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator =
    new OllamaApiClient(new Uri("http://localhost:11434"), "llama3.2:3b");//.EmbedAsync();
////1: generate a single embedding
//var embedding = await embeddingGenerator.GenerateVectorAsync("Hello, world!");
//Console.ForegroundColor = ConsoleColor.Green;
//Console.WriteLine($"Embeddding dimensions: {embedding.Span.Length}");
//foreach (var value in embedding.Span)
//{
//    Console.Write("{0:0.00}, ", value);
//}
//compare multiple embeddings using cosine similarity
var catVector = await embeddingGenerator.GenerateVectorAsync("cat");
var dogVector = await embeddingGenerator.GenerateVectorAsync("dog");
var kittenVector = await embeddingGenerator.GenerateVectorAsync("kitten");

Console.WriteLine($"cat-dog similarity: {TensorPrimitives.CosineSimilarity(catVector.Span, dogVector.Span):F2}");
Console.WriteLine($"cat-kitten similarity: {TensorPrimitives.CosineSimilarity(catVector.Span, kittenVector.Span):F2}");
Console.WriteLine($"dog-kitten similarity: {TensorPrimitives.CosineSimilarity(dogVector.Span, kittenVector.Span):F2}");