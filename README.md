# document-oriented-agent

Using Vector DB, Langchain and ChatGPT to get enhanced Responses and Cited Sources from Siteminder 'Help and learning' Document Database

### start application:

- copy `.env.example` to `.env` and add your OpenAI API key to .env file
- run `docker-compose up` to start container
- app will start on `localhost:1010`

### 

- Sample query:
`GET http://0.0.0.0:1010/query`
- Payload:

```
{
	"text": "<your question here>",
	"language": "<language>"
}
```
