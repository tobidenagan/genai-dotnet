using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel.Connectors.InMemory;
using OpenAI;
using System.ClientModel;
using VectorSearch;

//get credentials from user secrets
IConfigurationRoot config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
var credential = new ApiKeyCredential(config["GitHubModels:Token"] ?? throw new InvalidOperationException());
var options = new OpenAIClientOptions()
{
    Endpoint = new Uri("https://models.github.ai/inference")
};
//create an embedding generatot (text-embedding-3-small is an example)
IEmbeddingGenerator<string, Embedding<float>> generator =
    new OpenAIClient(credential, options)
    .GetEmbeddingClient("openai/text-embedding-3-small")
    .AsIEmbeddingGenerator();
//create and populate the vector store
var vectorStore = new InMemoryVectorStore();
var movieStore = vectorStore.GetCollection<int, Movie>("movies");
await movieStore.EnsureCollectionExistsAsync();
foreach (var movie in MovieData.Movies)
{
    //generate the embedding vector for the movie description
    movie.Vector = await generator.GenerateVectorAsync(movie.Description);
    //add the overall movie to the in-memory vector store;s movie collection
    await movieStore.UpsertAsync(movie);
}