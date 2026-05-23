# Data Directory

This directory stores the face recognition database and other data files.

## Face Database

- **File**: `face_database.pkl`
- **Format**: Python pickle file
- **Contents**: Dictionary mapping names to face embeddings

## Managing the Database

### Add Faces

```bash
# From image file
python3 examples/manage_database.py add "Person Name" path/to/photo.jpg

# From camera
python3 examples/manage_database.py capture "Person Name"
```

### View Database

```bash
python3 examples/manage_database.py list
```

### Remove Faces

```bash
python3 examples/manage_database.py remove "Person Name"
```

### Clear Database

```bash
python3 examples/manage_database.py clear
```

## Database Format

The database is a pickled Python dictionary:

```python
{
    "Person 1": [embedding1, embedding2, ...],
    "Person 2": [embedding1, embedding2, ...],
    ...
}
```

Each person can have multiple embeddings (from different photos/angles) to improve recognition accuracy.

## Backup

Regularly backup your database:

```bash
cp data/face_database.pkl data/face_database_backup_$(date +%Y%m%d).pkl
```

## Security

⚠️ **Important**: The database contains face embeddings which are considered biometric data. 

- Keep backups secure
- Consider encrypting the database file
- Follow privacy regulations (GDPR, CCPA, etc.)
- Obtain consent before storing face data
