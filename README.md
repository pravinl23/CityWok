# CityWok - Episode Identifier

## Backend Setup

### 1. Create Virtual Environment

```bash
cd backend
python3 -m venv venv
```

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Backend

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### 5. Deactivate Virtual Environment

When you're done:
```bash
deactivate
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

