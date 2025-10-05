﻿using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using OpenAI;
using System.ClientModel;

//get credentials from user secrets
IConfigurationRoot config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();
var credential = new ApiKeyCredential(config["GitHubModels:Token"] ?? throw new InvalidOperationException());
var options = new OpenAIClientOptions()
{
    Endpoint = new Uri("https://models.github.ai/inference")
};
//create a chat client
IChatClient client =
    new ChatClientBuilder(new OpenAIClient(credential, options).GetChatClient("openai/gpt-4o-mini").AsIChatClient())
    .UseFunctionInvocation()
    .Build();
var chatOptions = new ChatOptions
{
    Tools = [AIFunctionFactory.Create((string location, string unit) =>
    {
        //call to could be made here to a weather API to get the weather for the location
        var temperature = Random.Shared.Next(5, 20);
        var conditions = Random.Shared.Next(0, 1) == 0 ? "sunny" : "rainy";
        return $"The weather is {temperature} degrees C and {conditions}.";
    },
    name: "get_current_weather",
    description: "Get the current weather in a given location")]
};
List<ChatMessage> chatHistory = [new(ChatRole.User, """
    You are a hiking enthusiast who helps people discover fun hikes in their area.
    You're upbeat and friendly
    """)];
chatHistory.Add(new(ChatRole.User, """
    I live in Istanbul and I'm looking for a moderate intensity hike.
    What's the current weather like?
    """));
Console.WriteLine($"{chatHistory.Last().Role} >>> {chatHistory.Last()}");
ChatResponse response = await client.GetResponseAsync(chatHistory, chatOptions);
chatHistory.Add(new(ChatRole.Assistant, response.Text));
Console.WriteLine($"{chatHistory.Last().Role} >>> {chatHistory.Last()}");