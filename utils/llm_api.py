import requests
import os
import json
from typing import Dict, Optional

def get_huggingface_headers():
    """Get Hugging Face API headers"""
    api_key = os.getenv("HUGGINGFACE_API_KEY", "hf_default_key")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

def call_huggingface_api(model_name: str, prompt: str, max_length: int = 500) -> Optional[str]:
    """Call Hugging Face Inference API"""
    try:
        # Try the serverless inference API first
        url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = get_huggingface_headers()
        
        # Simple payload format
        payload = {"inputs": prompt}
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "")
                # Clean up the response
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                return generated
            elif isinstance(result, dict):
                generated = result.get("generated_text", "")
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                return generated
        else:
            print(f"HuggingFace API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling HuggingFace API: {str(e)}")
        return None

def generate_answer(question: str) -> Optional[str]:
    """Generate model answer for a given question using LLM"""
    try:
        # First try the actual LLM API
        llm_answer = call_free_llm_api(question)
        if llm_answer and len(llm_answer.strip()) > 20:
            return llm_answer
        
        # Fallback to knowledge base
        return get_intelligent_answer(question)
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return get_intelligent_answer(question)

def call_free_llm_api(question: str) -> Optional[str]:
    """Call free LLM APIs for answer generation"""
    try:
        # Try Groq API first (user has this key)
        groq_answer = call_groq_api(question)
        if groq_answer and len(groq_answer.strip()) > 20:
            return groq_answer.strip()
        
        # Try DeepSeek API as backup (if available)
        deepseek_answer = call_deepseek_api(question)
        if deepseek_answer and len(deepseek_answer.strip()) > 20:
            return deepseek_answer.strip()
        
        # Fallback to knowledge base only if APIs fail
        print("APIs unavailable, using knowledge base fallback")
        return get_intelligent_answer(question)
        
    except Exception as e:
        print(f"Error in free LLM API: {str(e)}")
        return get_intelligent_answer(question)

def call_deepseek_api(question: str) -> Optional[str]:
    """Call DeepSeek API - excellent free tier"""
    try:
        # Check for DeepSeek API key
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            print("DEEPSEEK_API_KEY not found, skipping DeepSeek API")
            return None
        
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {deepseek_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert educational assistant. Provide comprehensive, accurate, and well-structured answers to academic questions. Use clear explanations, examples, and organize your response with headings and bullet points when helpful."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nPlease provide a detailed, educational answer explaining key concepts, definitions, examples, and real-world applications."
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print("DeepSeek API success")
                return content.strip()
        else:
            print(f"DeepSeek API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling DeepSeek API: {str(e)}")
        return None

def call_groq_api(question: str) -> Optional[str]:
    """Call Groq API - fast and reliable free tier"""
    try:
        # Check for Groq API key
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("GROQ_API_KEY not found, skipping Groq API")
            return None
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert educational assistant. Provide comprehensive, accurate, and well-structured answers to academic questions."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nPlease provide a detailed educational answer with clear explanations and examples."
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print("Groq API success")
                return content.strip()
        else:
            print(f"Groq API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Groq API: {str(e)}")
        return None

def call_mistral_api(question: str) -> Optional[str]:
    """Call Mistral AI official free API"""
    try:
        # Check for Mistral API key
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            print("MISTRAL_API_KEY not found, skipping Mistral API")
            return None
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {mistral_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert educational assistant. Provide comprehensive, accurate, and well-structured answers to academic questions. Include key concepts, examples, and explanations."
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nPlease provide a detailed, educational answer covering the key concepts, definitions, examples, and significance of this topic."
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content.strip()
        else:
            print(f"Mistral API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Mistral API: {str(e)}")
        return None

def call_alternative_free_api(question: str) -> Optional[str]:
    """Try alternative free APIs"""
    try:
        # Use Hugging Face's text generation with a simpler approach
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key or api_key == "hf_default_key":
            return None
            
        # Try the Inference API with a simple text completion model
        url = "https://api-inference.huggingface.co/models/gpt2"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Create a more detailed prompt
        prompt = f"""Question: {question}

Provide a comprehensive, educational answer that covers:
- Key definitions and concepts
- Important details and explanations  
- Relevant examples
- Significance and applications

Answer:"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get("generated_text", "")
                # Clean up the response
                if "Answer:" in generated:
                    answer = generated.split("Answer:")[-1].strip()
                    if len(answer) > 20:
                        return answer
        
        return None
        
    except Exception as e:
        print(f"Alternative API error: {str(e)}")
        return None

def get_intelligent_answer(question: str) -> str:
    """Generate intelligent, specific answers based on question content"""
    try:
        question_lower = question.lower().strip()
        
        # Remove common question words to extract key topics
        question_words = question_lower.replace('?', '').split()
        content_words = [word for word in question_words if word not in 
                        ['what', 'is', 'are', 'how', 'why', 'when', 'where', 'who', 'which', 'does', 'do', 'can', 'will', 'would', 'should', 'could', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
        
        # Science Topics
        if any(word in question_lower for word in ['photosynthesis']):
            return """Photosynthesis is the biological process by which green plants, algae, and some bacteria convert light energy (usually from the sun) into chemical energy stored in glucose molecules.

The process occurs in two main stages:

1. **Light-dependent reactions (Photo phase)**:
   - Occur in the thylakoid membranes of chloroplasts
   - Chlorophyll absorbs light energy
   - Water molecules are split (H2O → 2H+ + 1/2O2 + 2e-)
   - ATP and NADPH are produced

2. **Light-independent reactions (Calvin Cycle)**:
   - Occur in the stroma of chloroplasts
   - CO2 is fixed into organic molecules
   - Uses ATP and NADPH from light reactions
   - Produces glucose (C6H12O6)

**Overall equation**: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2

**Significance**: Photosynthesis is essential for life on Earth as it produces oxygen and forms the base of food chains, converting inorganic carbon into organic compounds."""

        elif any(word in question_lower for word in ['mitosis', 'cell division']):
            return """Mitosis is the process of nuclear division in eukaryotic cells that produces two genetically identical diploid daughter cells from one parent cell.

**Phases of Mitosis:**

1. **Prophase**:
   - Chromatin condenses into visible chromosomes
   - Each chromosome consists of two sister chromatids
   - Nuclear envelope begins to break down
   - Centrioles move to opposite poles

2. **Metaphase**:
   - Chromosomes align at the cell's equatorial plate (metaphase plate)
   - Spindle fibers attach to kinetochores
   - Cell checkpoint ensures all chromosomes are properly attached

3. **Anaphase**:
   - Sister chromatids separate and move to opposite cell poles
   - Spindle fibers shorten, pulling chromatids apart

4. **Telophase**:
   - Nuclear envelopes reform around each set of chromosomes
   - Chromosomes begin to decondense
   - Spindle apparatus disassembles

**Cytokinesis** (division of cytoplasm) typically follows mitosis, completing cell division.

**Purpose**: Growth, tissue repair, and asexual reproduction in organisms."""

        elif any(word in question_lower for word in ['gravity', 'gravitational']):
            return """Gravity is a fundamental force of nature that causes objects with mass to attract each other. It is the weakest of the four fundamental forces but dominates at large scales.

**Newton's Law of Universal Gravitation:**
Every particle in the universe attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance between them.

**Formula**: F = G(m₁ × m₂)/r²

Where:
- F = gravitational force
- G = gravitational constant (6.674 × 10⁻¹¹ m³/kg⋅s²)
- m₁, m₂ = masses of the objects
- r = distance between centers of mass

**Earth's Gravity:**
- Acceleration due to gravity: 9.8 m/s² (approximately 10 m/s²)
- Objects fall at the same rate regardless of mass (in vacuum)
- Weight = mass × gravitational acceleration (W = mg)

**Einstein's General Relativity** describes gravity not as a force, but as the curvature of spacetime caused by mass and energy."""

        elif any(word in question_lower for word in ['democracy']):
            return """Democracy is a system of government where power ultimately rests with the people, either directly or through freely elected representatives.

**Key Principles:**

1. **Popular Sovereignty**: Government derives authority from the consent of the governed
2. **Political Equality**: All citizens have equal political rights and opportunities
3. **Majority Rule with Minority Rights**: Decisions made by majority vote, but minority rights protected
4. **Individual Liberty**: Protection of fundamental rights and freedoms

**Essential Features:**
- Free and fair elections with universal suffrage
- Multiple political parties and candidates
- Freedom of speech, press, and assembly
- Rule of law and independent judiciary
- Constitutional limits on government power
- Peaceful transfer of power

**Types:**
- **Direct Democracy**: Citizens vote directly on issues
- **Representative Democracy**: Citizens elect representatives to make decisions
- **Constitutional Democracy**: Democratic system with constitutional constraints

**Benefits**: Protects individual rights, promotes accountability, encourages civic participation, and generally leads to more stable and prosperous societies."""

        elif any(word in question_lower for word in ['water cycle', 'hydrologic']):
            return """The water cycle (hydrologic cycle) is the continuous movement of water through Earth's atmosphere, land, and oceans, driven by solar energy and gravity.

**Main Processes:**

1. **Evaporation**: Solar energy converts liquid water from oceans, lakes, and rivers into water vapor
2. **Transpiration**: Plants release water vapor through their leaves
3. **Condensation**: Water vapor cools and forms water droplets in clouds
4. **Precipitation**: Water falls as rain, snow, sleet, or hail
5. **Collection/Runoff**: Water flows back to oceans via rivers and streams
6. **Infiltration**: Water soaks into soil and becomes groundwater

**Energy Source**: The sun provides energy for evaporation and drives atmospheric circulation.

**Importance**:
- Distributes fresh water across Earth's surface
- Regulates global temperature
- Supports all life forms
- Shapes weather patterns and climate
- Replenishes freshwater resources

**Human Impact**: Pollution, deforestation, and climate change can disrupt natural water cycle patterns."""

        elif any(word in question_lower for word in ['ecosystem']):
            return """An ecosystem is a complex network of living organisms (biotic factors) interacting with each other and their physical environment (abiotic factors) in a specific area.

**Components:**

**Biotic Factors** (living):
- **Producers**: Plants and algae that make their own food through photosynthesis
- **Primary Consumers**: Herbivores that eat producers
- **Secondary Consumers**: Carnivores that eat primary consumers
- **Decomposers**: Bacteria and fungi that break down dead material

**Abiotic Factors** (non-living):
- Climate (temperature, rainfall)
- Soil composition
- Water availability
- Sunlight
- Air quality

**Energy Flow**:
- Energy enters through photosynthesis
- Flows through food chains and food webs
- Energy is lost as heat at each trophic level
- Only about 10% of energy transfers between levels

**Examples**: Forests, grasslands, deserts, wetlands, coral reefs, tundra

**Importance**: Ecosystems provide essential services like oxygen production, water purification, climate regulation, and food production."""

        elif any(word in question_lower for word in ['shakespeare']):
            return """William Shakespeare (1564-1616) was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language and the world's preeminent dramatist.

**Key Information:**
- Born in Stratford-upon-Avon, England
- Active during the Elizabethan and early Jacobean periods
- Wrote approximately 37 plays and 154 sonnets

**Major Works:**
- **Tragedies**: Hamlet, Macbeth, King Lear, Othello, Romeo and Juliet
- **Comedies**: A Midsummer Night's Dream, Much Ado About Nothing, The Tempest
- **Histories**: Henry V, Richard III, Julius Caesar

**Literary Contributions:**
- Expanded dramatic possibilities and psychological depth
- Created complex, multi-dimensional characters
- Invented thousands of words and phrases still used today
- Explored universal themes: love, power, betrayal, ambition, mortality

**Language Impact**: Added approximately 1,700 words to English language, including "assassination," "lonely," "generous," and "eyeball."

**Legacy**: His works continue to be performed, adapted, and studied worldwide, influencing literature, theater, and popular culture for over 400 years."""

        # Math topics
        elif any(word in question_lower for word in ['pythagorean theorem', 'pythagoras']):
            return """The Pythagorean Theorem is a fundamental principle in geometry that describes the relationship between the sides of a right triangle.

**Statement**: In a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides.

**Formula**: a² + b² = c²

Where:
- a and b are the lengths of the two legs (sides forming the right angle)
- c is the length of the hypotenuse (longest side)

**Example**: 
If a triangle has legs of 3 and 4 units:
3² + 4² = c²
9 + 16 = c²
25 = c²
c = 5 units

**Applications**:
- Construction and architecture
- Navigation and GPS systems
- Computer graphics and game development
- Engineering and design
- Distance calculations in coordinate geometry

**Historical Note**: Named after ancient Greek mathematician Pythagoras (c. 570-495 BCE), though the relationship was known to earlier civilizations including Babylonians and Indians."""

        elif any(word in question_lower for word in ['software engineering', 'software development']):
            return """Software engineering is a systematic, disciplined approach to the design, development, and maintenance of software systems. It applies engineering principles to software creation to ensure reliable, efficient, and maintainable solutions.

**Core Principles:**
- **Systematic approach**: Following structured methodologies and processes
- **Quality assurance**: Ensuring software meets requirements and standards
- **Project management**: Planning, organizing, and controlling software projects
- **Documentation**: Maintaining clear records of design decisions and code

**Key Phases of Software Development:**

1. **Requirements Analysis**: Understanding what the software needs to do
2. **System Design**: Creating the overall architecture and components
3. **Implementation**: Writing the actual code
4. **Testing**: Verifying the software works correctly
5. **Deployment**: Releasing the software to users
6. **Maintenance**: Ongoing support and updates

**Popular Methodologies:**
- **Waterfall**: Sequential phases with clear milestones
- **Agile**: Iterative development with continuous feedback
- **Scrum**: Framework for managing product development
- **DevOps**: Integration of development and operations

**Essential Skills:**
- Programming languages (Python, Java, C++, JavaScript, etc.)
- Database management
- Version control (Git)
- Testing frameworks
- System architecture
- Problem-solving and analytical thinking

**Career Opportunities:**
Software engineers work in various roles including web developer, mobile app developer, systems analyst, database administrator, and software architect across industries like technology, finance, healthcare, and entertainment.

**Importance**: Software engineering is crucial in our digital world, powering everything from smartphones and websites to medical devices and space exploration systems."""

        elif any(word in question_lower for word in ['artificial intelligence', 'machine learning', 'ai']):
            return """Artificial Intelligence (AI) is a branch of computer science focused on creating systems that can perform tasks typically requiring human intelligence, such as learning, reasoning, problem-solving, and decision-making.

**Types of AI:**

1. **Narrow AI (Weak AI)**: Designed for specific tasks
   - Examples: Voice assistants, recommendation systems, image recognition
   - Currently the most common form of AI

2. **General AI (Strong AI)**: Human-level intelligence across all domains
   - Theoretical concept not yet achieved
   - Would match human cognitive abilities

3. **Superintelligence**: AI that surpasses human intelligence
   - Hypothetical future development

**Key Subfields:**

**Machine Learning**: Systems that improve through experience
- **Supervised Learning**: Learning from labeled examples
- **Unsupervised Learning**: Finding patterns in unlabeled data
- **Reinforcement Learning**: Learning through rewards and penalties

**Deep Learning**: Neural networks with multiple layers
- Powers image recognition, natural language processing
- Used in self-driving cars, medical diagnosis

**Natural Language Processing (NLP)**: Understanding and generating human language
- Applications: Translation, chatbots, sentiment analysis

**Computer Vision**: Interpreting visual information
- Applications: Facial recognition, medical imaging, autonomous vehicles

**Current Applications:**
- Healthcare: Medical diagnosis, drug discovery
- Transportation: Self-driving cars, traffic optimization
- Finance: Fraud detection, algorithmic trading
- Entertainment: Content recommendation, game AI
- Business: Customer service chatbots, predictive analytics

**Challenges:**
- Ethical considerations and bias
- Job displacement concerns
- Privacy and security issues
- Need for large amounts of data
- Computational requirements

**Future Impact**: AI is expected to transform virtually every industry, potentially solving complex global challenges while raising important questions about human-AI collaboration."""

        elif any(word in question_lower for word in ['neural network', 'neural networks', 'neuralnetwork']):
            return """A neural network is a type of machine learning model inspired by the structure and functioning of the human brain. It's used to recognize patterns, make decisions, and predict outcomes from data.

**Basic Idea:**
A neural network is made up of layers of nodes (also called neurons), where:

• Each node takes inputs, applies weights, adds a bias, and passes the result through an activation function to produce an output
• These outputs become inputs for the next layer
• The network learns by adjusting weights and biases based on training data

**Structure:**

**Input Layer** – Receives the raw data (e.g., pixels of an image, features of a dataset)

**Hidden Layers** – Do the computations and transformations (can be one or many layers)
- Process information through mathematical operations
- Extract increasingly complex features
- Each layer builds upon the previous layer's outputs

**Output Layer** – Produces the final result (e.g., predicted class, probability, or numeric value)

**How It Learns:**

1. **Forward Pass**: The network makes predictions by passing data through all layers
2. **Error Calculation**: Compares predictions to actual results (calculates loss/error)
3. **Backpropagation**: Adjusts weights and biases using algorithms like gradient descent to reduce error
4. **Iteration**: Repeats this process thousands of times to improve accuracy

**Key Components:**

**Weights and Biases**: Parameters that determine how strongly connections influence outputs
**Activation Functions**: Mathematical functions (like ReLU, sigmoid, tanh) that introduce non-linearity
**Loss Functions**: Measure how far predictions are from actual values
**Optimizers**: Algorithms (like Adam, SGD) that update weights efficiently

**Types of Neural Networks:**

**Feedforward Networks**: Information flows in one direction
**Convolutional Neural Networks (CNNs)**: Specialized for image processing
**Recurrent Neural Networks (RNNs)**: Handle sequential data like text or time series
**Long Short-Term Memory (LSTM)**: Advanced RNNs that remember long-term patterns

**Real-World Applications:**

• **Image and speech recognition**: Face detection, medical imaging, voice assistants
• **Natural language processing**: Translation, chatbots, sentiment analysis
• **Autonomous vehicles**: Object detection, path planning
• **Recommendation systems**: Netflix, Amazon, Spotify suggestions
• **Finance**: Fraud detection, algorithmic trading
• **Healthcare**: Drug discovery, diagnosis assistance
• **Gaming**: AI opponents, procedural content generation

**Advantages:**
- Can learn complex patterns from large datasets
- Highly flexible and adaptable
- Excellent for tasks involving unstructured data
- Can approximate virtually any function

**Challenges:**
- Requires large amounts of training data
- Computationally intensive
- "Black box" - difficult to interpret decisions
- Can overfit to training data
- Sensitive to hyperparameter choices

**Simple Analogy:**
Think of a neural network like a team of people making a decision. Each person (node) receives information, processes it based on their expertise (weights), and passes their conclusion to the next group. The final decision emerges from this collaborative process, and the team gets better over time by learning from their mistakes.

Let me know if you want a diagram or more details about any specific aspect!"""

        elif any(word in question_lower for word in ['climate change', 'global warming']):
            return """Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the Industrial Revolution.

**Causes:**

**Greenhouse Gases**: Trap heat in Earth's atmosphere
- **Carbon Dioxide (CO₂)**: From burning fossil fuels (75% of emissions)
- **Methane (CH₄)**: From agriculture and landfills
- **Nitrous Oxide (N₂O)**: From fertilizers and industrial processes
- **Fluorinated Gases**: From refrigeration and industrial applications

**Human Activities:**
- Burning fossil fuels for electricity, heat, and transportation
- Deforestation reducing CO₂ absorption
- Industrial processes and manufacturing
- Agriculture, especially livestock farming

**Evidence:**
- Global average temperature has risen 1.1°C since 1880
- Arctic sea ice declining by 13% per decade
- Rising sea levels (21-24 cm since 1880)
- Changing precipitation patterns
- More frequent extreme weather events

**Impacts:**

**Environmental:**
- Melting ice caps and glaciers
- Ocean acidification
- Species extinction and habitat loss
- Coral reef bleaching

**Human Society:**
- Food security threats
- Water scarcity
- Displacement of populations
- Economic costs from extreme weather
- Health impacts from heat and disease

**Solutions:**

**Mitigation** (Reducing emissions):
- Renewable energy (solar, wind, hydro)
- Energy efficiency improvements
- Electric vehicles and public transportation
- Carbon pricing and regulations
- Reforestation and afforestation

**Adaptation** (Adjusting to changes):
- Climate-resilient infrastructure
- Drought-resistant crops
- Coastal protection measures
- Emergency preparedness systems

**International Efforts:**
- Paris Agreement (2015): Global commitment to limit warming
- IPCC Reports: Scientific assessments of climate change
- National commitments to reduce emissions

**Individual Actions:**
- Reduce energy consumption
- Use sustainable transportation
- Support renewable energy
- Make conscious consumer choices
- Advocate for policy changes

**Urgency**: Scientists emphasize that rapid action is needed to limit warming to 1.5°C and avoid the most catastrophic impacts."""

        elif any(word in question_lower for word in ['world war', 'ww1', 'ww2', 'first world war', 'second world war']):
            if any(word in question_lower for word in ['first', 'ww1', '1']):
                return """World War I (1914-1918) was a global conflict that fundamentally changed the political landscape of the 20th century.

**Causes:**
- **Imperialism**: Competition for colonies and global influence
- **Alliance System**: Complex web of mutual defense treaties
- **Nationalism**: Ethnic tensions, especially in the Balkans
- **Militarism**: Arms race and military buildup
- **Immediate Trigger**: Assassination of Archduke Franz Ferdinand (June 28, 1914)

**Major Powers:**
- **Allied Powers**: Britain, France, Russia, later joined by United States
- **Central Powers**: Germany, Austria-Hungary, Ottoman Empire, Bulgaria

**Key Features:**
- **Trench Warfare**: Static battle lines with horrific conditions
- **New Technology**: Machine guns, poison gas, tanks, aircraft
- **Total War**: Entire societies mobilized for war effort
- **Global Scale**: Fighting in Europe, Africa, Middle East, and at sea

**Major Battles:**
- Battle of the Somme (1916): Over 1 million casualties
- Battle of Verdun (1916): Longest single battle
- Battle of Passchendaele (1917): Epitomized trench warfare horrors

**Consequences:**
- **Human Cost**: 15-20 million deaths, millions wounded
- **Political Changes**: Fall of German, Austro-Hungarian, Russian, and Ottoman empires
- **Treaty of Versailles (1919)**: Harsh terms for Germany, war guilt clause
- **New Nations**: Poland, Czechoslovakia, Yugoslavia created
- **Economic Impact**: Massive debt, inflation, economic instability
- **Social Changes**: Women's roles expanded, class structures shifted

**Long-term Impact**: Set stage for World War II, Russian Revolution, and reshaped global politics."""
            else:
                return """World War II (1939-1945) was the deadliest and most widespread conflict in human history, involving most of the world's nations.

**Causes:**
- **Treaty of Versailles**: Harsh terms created resentment in Germany
- **Rise of Fascism**: Hitler (Germany), Mussolini (Italy), militarism in Japan
- **Economic Depression**: Global economic crisis enabled extremist movements
- **Appeasement Policy**: Failed attempts to avoid war through concessions
- **Immediate Trigger**: Germany's invasion of Poland (September 1, 1939)

**Major Powers:**
- **Axis Powers**: Germany, Italy, Japan
- **Allied Powers**: Britain, Soviet Union, United States, China, France

**Key Phases:**
1. **Early German Success (1939-1941)**: Blitzkrieg tactics, conquest of Europe
2. **Global Expansion (1941-1942)**: Germany invades USSR, Japan attacks Pearl Harbor
3. **Turning Point (1942-1943)**: Stalingrad, Midway, North Africa campaigns
4. **Allied Victory (1944-1945)**: D-Day invasion, liberation of Europe, atomic bombs

**Major Theaters:**
- **European Theater**: Holocaust, Eastern Front, Western Front
- **Pacific Theater**: Island-hopping campaign, naval warfare
- **North African Theater**: Desert warfare, control of Suez Canal

**Consequences:**
- **Human Cost**: 70-85 million deaths, including 6 million Jews in Holocaust
- **Political Reorganization**: United Nations formed, Cold War begins
- **Decolonization**: Weakened European powers lose colonies
- **Nuclear Age**: Atomic weapons change warfare forever
- **Economic Changes**: US emerges as superpower, Marshall Plan rebuilds Europe

**Legacy**: Established international law, human rights concepts, and shaped modern geopolitics."""

        elif any(word in question_lower for word in ['periodic table', 'elements', 'mendeleev']):
            return """The Periodic Table is a systematic arrangement of chemical elements organized by their atomic number, electron configuration, and recurring chemical properties.

**Organization:**
- **Rows (Periods)**: Elements with the same number of electron shells
- **Columns (Groups/Families)**: Elements with similar chemical properties
- **Atomic Number**: Number of protons in nucleus (increases left to right)

**Key Groups:**
1. **Alkali Metals (Group 1)**: Highly reactive metals (Li, Na, K)
2. **Alkaline Earth Metals (Group 2)**: Reactive metals (Mg, Ca, Ba)
3. **Halogens (Group 17)**: Reactive nonmetals (F, Cl, Br, I)
4. **Noble Gases (Group 18)**: Unreactive gases (He, Ne, Ar, Kr)
5. **Transition Metals**: Most metals, variable oxidation states

**Periodic Trends:**
- **Atomic Radius**: Decreases across period, increases down group
- **Ionization Energy**: Increases across period, decreases down group
- **Electronegativity**: Increases across period, decreases down group
- **Metallic Character**: Decreases across period, increases down group

**Historical Development:**
- **Dmitri Mendeleev (1869)**: Created first periodic table, predicted unknown elements
- **Henry Moseley (1913)**: Arranged by atomic number instead of atomic mass
- **Modern Form**: Incorporates quantum mechanical understanding

**Applications:**
- Predicting chemical behavior and reactions
- Understanding bonding patterns
- Materials science and engineering
- Medical and pharmaceutical research
- Nuclear physics and chemistry

**Current Status**: 118 known elements, with new superheavy elements still being discovered and studied."""

        elif any(word in question_lower for word in ['dna', 'genetics', 'heredity']):
            return """DNA (Deoxyribonucleic Acid) is the hereditary material that contains genetic instructions for the development, functioning, and reproduction of all known living organisms.

**Structure:**
- **Double Helix**: Two complementary strands twisted around each other
- **Nucleotides**: Building blocks containing sugar (deoxyribose), phosphate, and nitrogenous base
- **Base Pairs**: Adenine-Thymine (A-T) and Guanine-Cytosine (G-C)
- **Antiparallel Strands**: Run in opposite directions (5' to 3')

**Functions:**
1. **Information Storage**: Contains genetic code in sequence of bases
2. **Inheritance**: Passes genetic information to offspring
3. **Protein Synthesis**: Provides instructions for making proteins
4. **Cell Division**: Replicates to ensure each cell has complete genetic information

**Genetic Code:**
- **Triplet Code**: Three bases (codon) specify one amino acid
- **64 Codons**: Code for 20 amino acids plus start/stop signals
- **Universal**: Same code used by virtually all organisms

**DNA Replication:**
1. **Unwinding**: Double helix separates at replication fork
2. **Priming**: RNA primers provide starting point
3. **Synthesis**: DNA polymerase adds complementary nucleotides
4. **Proofreading**: Enzymes check and correct errors

**Protein Synthesis:**
1. **Transcription**: DNA → RNA in nucleus
2. **Translation**: RNA → Protein at ribosomes
3. **Gene Expression**: Regulation of when genes are active

**Applications:**
- **Medicine**: Gene therapy, personalized medicine, disease diagnosis
- **Forensics**: DNA fingerprinting, paternity testing
- **Agriculture**: Genetic modification, crop improvement
- **Evolution**: Understanding relationships between species
- **Biotechnology**: Cloning, genetic engineering

**Human Genome**: ~3 billion base pairs, ~20,000-25,000 genes, completed sequencing in 2003."""

        # Try to extract key concepts and provide relevant information
        elif content_words:
            main_topic = ' '.join(content_words[:2])
            return f"""**Answer for: {question}**

To properly answer this question about {main_topic}, consider the following key points:

**Definition/Overview:**
{main_topic.title()} refers to [the fundamental concept or process being asked about]. This is important in [relevant field/context] because [significance].

**Key Characteristics:**
• [Primary feature or aspect]
• [Secondary feature or aspect]  
• [Additional important point]

**Examples/Applications:**
• [Concrete example 1]
• [Concrete example 2]
• [Real-world application]

**Related Concepts:**
{main_topic.title()} connects to [related topics] and influences [broader implications].

**Significance:**
Understanding {main_topic} is important because [practical relevance and why it matters].

*Note: This is a template answer. For a complete response, specific details about {main_topic} should be researched and included.*"""

        else:
            return f"""**Model Answer for: "{question}"**

This question requires a comprehensive response covering:

**Introduction:**
Provide context and define key terms mentioned in the question.

**Main Content:**
• Address each part of the question systematically
• Include specific facts, examples, and evidence
• Explain relationships between concepts
• Demonstrate deep understanding of the topic

**Analysis:**
• Discuss implications and significance
• Consider different perspectives where relevant
• Connect to broader themes or concepts

**Conclusion:**
Summarize key points and reinforce the main argument or explanation.

**Additional Notes:**
Ensure your answer is well-structured, uses appropriate terminology, and provides specific examples to support your points."""
            
    except Exception as e:
        return f"Error generating answer: {str(e)}. Please provide a comprehensive answer addressing all aspects of the question."

def create_template_answer(question: str) -> str:
    """Create a helpful template answer for any question"""
    try:
        question_lower = question.lower()
        
        # Try to create more specific answers based on common academic topics
        if any(word in question_lower for word in ['photosynthesis', 'plants', 'chlorophyll']):
            return """Photosynthesis is the process by which green plants and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose.

The process occurs in two main stages:
1. Light-dependent reactions (in thylakoids): Light energy is captured and converted to ATP and NADPH
2. Light-independent reactions (Calvin cycle): CO2 is fixed into glucose using ATP and NADPH

The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2

This process is essential for life on Earth as it produces oxygen and forms the base of most food chains."""

        elif any(word in question_lower for word in ['mitosis', 'cell division', 'chromosomes']):
            return """Mitosis is the process of cell division that produces two genetically identical diploid cells from one parent cell.

The phases of mitosis are:
1. Prophase: Chromosomes condense and become visible, nuclear envelope breaks down
2. Metaphase: Chromosomes align at the cell's equator
3. Anaphase: Sister chromatids separate and move to opposite poles
4. Telophase: Nuclear envelopes reform around each set of chromosomes

Cytokinesis then divides the cytoplasm, completing cell division. This process is essential for growth, repair, and asexual reproduction."""

        elif any(word in question_lower for word in ['gravity', 'newton', 'force', 'acceleration']):
            return """Gravity is a fundamental force of attraction between all objects with mass. Newton's law of universal gravitation states that every particle attracts every other particle with a force proportional to the product of their masses and inversely proportional to the square of the distance between them.

The formula is: F = G(m1×m2)/r²

Where:
- F is the gravitational force
- G is the gravitational constant
- m1 and m2 are the masses of the objects
- r is the distance between centers of mass

On Earth, gravity gives objects an acceleration of approximately 9.8 m/s² toward the center of the planet."""

        elif any(word in question_lower for word in ['democracy', 'government', 'voting', 'elections']):
            return """Democracy is a system of government where power is held by the people, either directly or through elected representatives.

Key features of democracy include:
1. Free and fair elections with universal suffrage
2. Protection of individual rights and freedoms
3. Rule of law and constitutional governance
4. Separation of powers between branches of government
5. Political pluralism and competition

Democratic systems ensure accountability through regular elections, checks and balances, and protection of minority rights while allowing majority rule."""

        elif any(word in question_lower for word in ['what', 'define', 'definition', 'meaning']):
            # Extract key terms from the question for more specific answers
            key_terms = [word for word in question_lower.split() if len(word) > 3 and word not in ['what', 'define', 'definition', 'meaning', 'explain', 'describe']]
            main_topic = key_terms[0] if key_terms else "the concept"
            
            return f"""To answer "what is {main_topic}":

{main_topic.title()} refers to [provide clear definition]. 

Key characteristics include:
• [First important feature or aspect]
• [Second important feature or aspect]  
• [Third important feature or aspect]

This concept is significant because [explain importance and applications].

Examples of {main_topic} include: [provide specific, relevant examples].

{main_topic.title()} relates to [broader context or connected concepts]."""

        elif any(word in question_lower for word in ['how', 'process', 'steps', 'method']):
            # Extract the main process from the question
            process_words = [word for word in question_lower.split() if word not in ['how', 'does', 'do', 'work', 'process', 'steps', 'method']]
            main_process = ' '.join(process_words[:3]) if process_words else "this process"
            
            return f"""The process of {main_process} involves several key steps:

Step 1: [Initial stage - describe what happens first]
Step 2: [Second stage - explain the next phase]
Step 3: [Third stage - detail the following step]
Step 4: [Final stage - describe the conclusion]

Important factors that affect {main_process}:
• [First influencing factor]
• [Second influencing factor]
• [Third influencing factor]

The significance of understanding {main_process} is [explain why this knowledge matters]."""

        elif any(word in question_lower for word in ['why', 'reason', 'cause', 'because']):
            # Extract the phenomenon being questioned
            topic_words = [word for word in question_lower.split() if word not in ['why', 'does', 'do', 'is', 'are', 'reason', 'cause', 'because']]
            main_topic = ' '.join(topic_words[:3]) if topic_words else "this phenomenon"
            
            return f"""The reasons for {main_topic} are:

Primary reasons:
1. [First major cause or reason - explain in detail]
2. [Second major cause or reason - provide explanation]
3. [Third major cause or reason - give context]

Contributing factors:
• [Additional factor 1]
• [Additional factor 2]
• [Additional factor 3]

The consequences of {main_topic} include [explain effects and implications].

Understanding these reasons helps us [explain practical applications or benefits]."""

        else:
            # Generic but still specific to the question
            return f"""Model answer for: "{question}"

Introduction:
[Provide a brief overview of the topic and its significance]

Main Points:
1. [First key concept or argument related to the question]
   - Supporting details and examples
   - Explanation of importance

2. [Second key concept or argument]
   - Relevant information and context
   - Connection to the broader topic

3. [Third key concept or argument]
   - Specific examples or applications
   - Analysis and implications

Conclusion:
[Summarize the key points and explain the overall significance of the topic]

This answer should demonstrate understanding of [identify the main subject area] and provide specific, detailed information that directly addresses the question asked."""
            
    except Exception as e:
        return f"""Model answer for: "{question}"

[Provide a comprehensive response that directly addresses this specific question, including relevant facts, examples, and explanations. Structure your answer with clear introduction, main points, and conclusion.]"""

def grade_answer(question: str, model_answer: str, student_answer: str, max_marks: int) -> Dict:
    """Grade student answer using LLM (no rubric)"""
    try:
        llm_result = call_llm_for_grading(question, model_answer, student_answer, max_marks)
        if llm_result:
            return parse_grading_response(llm_result.get('raw_response', llm_result.get('feedback', '')), max_marks)
        return basic_grade_answer(question, model_answer, student_answer, max_marks)
    except Exception as e:
        print(f"Error grading answer: {str(e)}")
        return basic_grade_answer(question, model_answer, student_answer, max_marks)

def call_llm_for_grading(question: str, model_answer: str, student_answer: str, max_marks: int) -> Optional[Dict]:
    """Use LLM to grade student answers"""
    try:
        # Try Groq API for grading first (user has this key)
        groq_result = call_groq_for_grading(question, model_answer, student_answer, max_marks)
        if groq_result:
            return groq_result
        
        # Try DeepSeek API as backup
        deepseek_result = call_deepseek_for_grading(question, model_answer, student_answer, max_marks)
        if deepseek_result:
            return deepseek_result
        
        return None
        
    except Exception as e:
        print(f"Error in LLM grading: {str(e)}")
        return None

def call_deepseek_for_grading(question: str, model_answer: str, student_answer: str, max_marks: int) -> Optional[Dict]:
    """Use DeepSeek API to grade student answers"""
    try:
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_key:
            print("DEEPSEEK_API_KEY not found for grading")
            return None
        
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {deepseek_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert teacher grading student answers. Provide fair, detailed feedback with specific scores based on accuracy, completeness, and understanding."
                },
                {
                    "role": "user",
                    "content": f"""Grade this student answer:

Question: {question}

Model Answer: {model_answer if model_answer else "No model answer provided"}

Student Answer: {student_answer}

Grading Criteria:
- Accuracy and correctness (40%)
- Completeness of answer (30%)
- Understanding demonstrated (20%)
- Clarity of explanation (10%)

IMPORTANT: You MUST provide the following rubric breakdown, with each line present, before the total score and feedback. Use this exact format:
Rubric:
Accuracy: [score out of 2]
Completeness: [score out of 2]
Understanding: [score out of 1]
Clarity: [score out of 1]
Score: [number out of {max_marks}]
Feedback: [detailed explanation]"""
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return parse_grading_response(content, max_marks)
        else:
            print(f"DeepSeek grading API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling DeepSeek for grading: {str(e)}")
        return None

def call_groq_for_grading(question: str, model_answer: str, student_answer: str, max_marks: int) -> Optional[Dict]:
    """Use Groq API to grade student answers"""
    try:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("GROQ_API_KEY not found for grading")
            return None
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert teacher grading student answers. Provide fair, detailed feedback with specific scores."
                },
                {
                    "role": "user",
                    "content": f"""Grade this student answer:

Question: {question}
Model Answer: {model_answer if model_answer else "No model answer provided"}
Student Answer: {student_answer}

Provide your response in exactly this format:
Rubric:
Accuracy: [score out of X]
Completeness: [score out of X]
Understanding: [score out of X]
Clarity: [score out of X]
Score: [number out of {max_marks}]
Feedback: [detailed explanation]"""
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return parse_grading_response(content, max_marks)
        else:
            print(f"Groq grading API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Groq for grading: {str(e)}")
        return None

def call_mistral_for_grading(question: str, model_answer: str, student_answer: str, max_marks: int) -> Optional[Dict]:
    """Use Mistral API to grade student answers"""
    try:
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_key:
            print("MISTRAL_API_KEY not found for grading")
            return None
        
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {mistral_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert teacher grading student answers. Provide fair, detailed feedback with specific scores."
                },
                {
                    "role": "user",
                    "content": f"""Grade this student answer:

Question: {question}

Model Answer: {model_answer if model_answer else "No model answer provided"}

Student Answer: {student_answer}

Grading Criteria:
- Accuracy and correctness (40%)
- Completeness of answer (30%)
- Understanding demonstrated (20%)
- Clarity of explanation (10%)

Provide your response in exactly this format:
Rubric:
Accuracy: [score out of X]
Completeness: [score out of X]
Understanding: [score out of X]
Clarity: [score out of X]
Score: [number out of {max_marks}]
Feedback: [detailed explanation]"""
                }
            ],
            "max_tokens": 400,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return parse_grading_response(content, max_marks)
        else:
            print(f"Mistral grading API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Mistral for grading: {str(e)}")
        return None

def parse_grading_response(response: str, max_marks: int) -> Dict:
    """Parse LLM grading response (no rubric)"""
    try:
        import re
        # Extract total score
        score_match = re.search(r'Score\s*:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        score = 0
        if score_match:
            score = min(float(score_match.group(1)), max_marks)
        # Extract feedback
        feedback_match = re.search(r'Feedback\s*:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
        feedback = "No detailed feedback provided"
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        if score == 0:
            number_match = re.search(r'(\d+(?:\.\d+)?)', response)
            if number_match:
                score = min(float(number_match.group(1)), max_marks)
        return {
            'score': score,
            'feedback': feedback,
            'max_marks': max_marks
        }
    except Exception as e:
        return {
            'score': max_marks * 0.5,
            'feedback': f"Error parsing grading response: {str(e)}",
            'max_marks': max_marks
        }

def basic_grade_answer(question: str, model_answer: str, student_answer: str, max_marks: int) -> Dict:
    """Basic fallback grading when AI is not available, using rubric internally (not displayed)"""
    try:
        student_answer = student_answer.strip()
        # Heuristic rubric: length for completeness, keyword overlap for accuracy, etc.
        rubric = {'Accuracy': 0, 'Completeness': 0, 'Understanding': 0, 'Clarity': 0}
        if len(student_answer) == 0:
            feedback = "No answer provided."
            return {'score': 0, 'feedback': feedback + " (Basic grading used - AI grading unavailable)", 'max_marks': max_marks}
        # Completeness (length)
        if len(student_answer) > 100:
            rubric['Completeness'] = 2
        elif len(student_answer) > 50:
            rubric['Completeness'] = 1.5
        elif len(student_answer) > 10:
            rubric['Completeness'] = 1
        # Accuracy (keyword overlap)
        if model_answer and len(model_answer.strip()) > 0:
            model_words = set(model_answer.lower().split())
            student_words = set(student_answer.lower().split())
            if len(model_words) > 0:
                overlap = len(model_words.intersection(student_words)) / len(model_words)
                if overlap > 0.5:
                    rubric['Accuracy'] = 2
                elif overlap > 0.2:
                    rubric['Accuracy'] = 1
        # Understanding (if answer is not just copied, and covers more than 1 sentence)
        if len(student_answer.split('.')) > 1:
            rubric['Understanding'] = 1
        if len(student_answer.split('.')) > 2:
            rubric['Understanding'] = 2
        # Clarity (basic: if answer is not a single run-on sentence)
        if len(student_answer.split('.')) > 1:
            rubric['Clarity'] = 1
        if len(student_answer.split('.')) > 2:
            rubric['Clarity'] = 1.5
        # Total score: sum rubric (max 7.5), scale to max_marks
        raw_score = sum(rubric.values())
        score = round((raw_score / 7.5) * max_marks, 1)
        feedback = f"(Basic grading used - AI grading unavailable)"
        return {'score': score, 'feedback': feedback, 'max_marks': max_marks}
    except Exception as e:
        return {'score': max_marks * 0.5, 'feedback': f"Error in grading: {str(e)}. Manual review recommended.", 'max_marks': max_marks}

def get_answer_similarity(answer1: str, answer2: str) -> float:
    """Calculate basic similarity between two answers (fallback method)"""
    try:
        if not answer1 or not answer2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0
