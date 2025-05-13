import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def create_drive_folder(service, folder_name):
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    file = service.files().create(body=file_metadata, fields='id').execute()
    folder_id = file.get('id')

    # Share the folder (optional)
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(fileId=folder_id, body=permission).execute()

    # Print sharable link
    print(f'Folder created: https://drive.google.com/drive/folders/{folder_id}')
    return folder_id


def upload_file(service, file_path, parent_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [parent_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f'Uploaded {file_path} → File ID: {file.get("id")}')

def main():
    service = authenticate()

    # STEP 1: Create a folder on Drive
    folder_id = create_drive_folder(service, 'face_dataset')

    # STEP 2: Upload all files from local face_dataset/
    local_folder = 'face_dataset'
    for filename in os.listdir(local_folder):
        file_path = os.path.join(local_folder, filename)
        if os.path.isfile(file_path):
            upload_file(service, file_path, folder_id)

if __name__ == '__main__':
    main()
