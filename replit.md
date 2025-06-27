# Assessment Platform

## Overview

A full-stack web application built with Streamlit that enables teachers to create subjective assessments and provides AI-powered grading for student submissions. The platform uses Large Language Models (LLMs) via Hugging Face API for answer generation and evaluation, along with plagiarism detection capabilities.

## System Architecture

The application follows a simple single-page application (SPA) architecture using Streamlit as both the frontend and backend framework:

- **Frontend**: Streamlit provides the web interface with sidebar navigation
- **Backend**: Python-based processing with utility modules for external API integrations
- **Data Storage**: JSON file-based persistence for assessments and student responses
- **External APIs**: Hugging Face for LLM operations, RapidAPI for plagiarism detection

## Key Components

### Core Application (`app.py`)
- Main entry point with Streamlit interface
- Session state management for assessments and student answers
- Dual-mode functionality: Create Assessment (Teacher) and Attempt Assessment (Student)
- JSON-based data persistence with automatic save/load functionality

### LLM Integration (`utils/llm_api.py`)
- Multi-model Hugging Face API integration with fallback strategy
- Comprehensive knowledge base covering major academic subjects
- Primary models: FLAN-T5, DialoGPT, Blenderbot for better coverage
- Answer generation for teacher-created questions across various subjects
- AI-based grading system with detailed feedback parsing
- Fallback to curated knowledge base for reliable answers

### Plagiarism Detection (`utils/plag_checker.py`)
- RapidAPI integration for plagiarism checking
- Multi-language support with citation detection capabilities
- Percentage-based similarity scoring
- Source detection and reporting

### Data Storage (`assessments.json`)
- Simple JSON structure for storing assessments
- Timestamp tracking for last updates
- In-memory session state with file persistence

## Data Flow

1. **Assessment Creation Flow**:
   - Teacher enters question details
   - Optional model answer generation via LLM API
   - Assessment saved to JSON file with unique identifier
   - Running total calculation for marks

2. **Assessment Attempt Flow**:
   - Student selects from available assessments
   - Questions presented sequentially with copy-paste prevention
   - Answers submitted for AI grading
   - Plagiarism check performed on submissions
   - Results calculated and presented

3. **Grading Process**:
   - Student answer compared against model answer using LLM
   - Relevance and accuracy scoring
   - Plagiarism percentage calculation
   - Combined scoring with detailed feedback

## External Dependencies

### Primary Dependencies
- **Streamlit (v1.46.1+)**: Web framework for UI and backend
- **Requests (v2.32.4+)**: HTTP client for API calls

### Additional Packages
- **Altair**: Data visualization (included with Streamlit)
- **Cachetools**: API response caching
- **Attrs**: Enhanced class definitions

### External APIs
- **Hugging Face Inference API**: LLM services (Mistral-7B-Instruct)
- **RapidAPI Plagiarism Service**: Text similarity detection

### Environment Variables
- `HUGGINGFACE_API_KEY`: Authentication for Hugging Face services
- `RAPIDAPI_KEY`: Authentication for plagiarism detection service

## Deployment Strategy

### Replit Configuration
- **Target**: Autoscale deployment
- **Runtime**: Python 3.11 with Nix package management
- **Port**: 5000 (configured for external access)
- **Startup Command**: `streamlit run app.py --server.port 5000`

### Server Configuration
- Headless mode enabled for production deployment
- Cross-origin access configured for 0.0.0.0
- Custom port binding for cloud deployment compatibility

### Scaling Considerations
- Stateless design allows horizontal scaling
- JSON file storage suitable for small-medium scale
- API rate limiting may require caching strategies
- Session state isolation ensures multi-user support

## Changelog

```
Changelog:
- June 27, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

## Technical Notes

- The application currently uses file-based JSON storage but is designed to easily migrate to database solutions like PostgreSQL with Drizzle ORM
- Plagiarism detection includes multiple fallback methods for different API response formats
- LLM integration supports model switching and parameter tuning
- Copy-paste prevention uses client-side warnings (JavaScript implementation pending)
- Error handling includes graceful degradation for API failures