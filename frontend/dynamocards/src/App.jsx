import React, { useState } from 'react';
import axios from 'axios';
import Flashcard from './Flashcard.jsx';
import './Flashcard.css'

function App(){
  const [youtubeLink, setYoutubeLink] = useState("");
  const [keyConcepts, setKeyConcepts] = useState([]);
  const handleLinkChange = (event) => {
    setYoutubeLink(event.target.value);
  };
  const sendLink  = async () => {
    try {
      const response = await axios.post("http://localhost:8000/analyze_video", {
        youtube_link : youtubeLink,
     });
     const data = response.data;
     if(data.keyConcepts && Array.isArray(data.key_concepts)){
      setKeyConcepts(data.key_concepts);
     }
     else{
      console.error("Data does not contain Key Concepts :",data);
     }
    } catch(error){
      console.log(error);
    }
  };

  const discardFlashcard = (index) => {
    setKeyConcepts(currentConcepts => currentConcepts.filter((_, i) => i != index));
  }
 
  return (
    <div className="App">
      <h1>Youtube Link to FlashCards Generator</h1>
      <div className='inputContainer'>
      <input 
        type="text" 
        placeholder="Paste your youtube Link" 
        value={youtubeLink} 
        onChange={handleLinkChange}
        className='inputField'
      />
      <button onClick={sendLink}>
        Generate FlashCards
      </button>
      </div>
      <div className='flashCardsContainer'>
        {keyConcepts.map((concept,index) => {
          <FlashCard
            key = {index}
            term = {concept.term}
            definition = {concept.definition}
            onDiscard = {() => discardFlashcard(index)}
            ></FlashCard>
        })}
      </div>
    </div>
  );
  
}
export default App;