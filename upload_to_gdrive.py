import os.path
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# --- Configuration ---
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# Path to the file you want to upload
FILE_TO_UPLOAD = 'weights-and-history.zip'
# Name of the file as it will appear in Google Drive
DRIVE_FILENAME = 'weights-and-history.zip'
# Path to store the authentication token
TOKEN_PATH = 'token.json'
# Path to your downloaded OAuth 2.0 credentials
CREDENTIALS_PATH = 'client_secret_1095028420845-kqe7klstsrus2o0bb4a05n7ms2eofp46.apps.googleusercontent.com.json'
# --- End Configuration ---

def authenticate():
    """Handles authentication and returns the Drive API service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists(TOKEN_PATH):
        try:
            # Use pickle for compatibility with older quickstart examples
            # Consider using google.oauth2.credentials.Credentials.from_authorized_user_file
            with open(TOKEN_PATH, 'rb') as token:
                 creds = pickle.load(token)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
             print(f"Error loading {TOKEN_PATH}. Attempting re-authentication.")
             creds = None # Force re-authentication if token file is corrupted

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}. Need to re-authenticate.")
                # Delete potentially corrupted token file
                if os.path.exists(TOKEN_PATH):
                    os.remove(TOKEN_PATH)
                creds = None # Force re-authentication
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                print(f"Error: Credentials file not found at '{CREDENTIALS_PATH}'")
                print("Please download your OAuth 2.0 credentials from Google Cloud Console")
                print("and save it as 'credentials.json' in the script's directory.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_PATH, SCOPES)
                # Use run_local_server instead of run_console for better UX
                creds = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Error during authentication flow: {e}")
                return None
        # Save the credentials for the next run using pickle
        try:
            with open(TOKEN_PATH, 'wb') as token:
                pickle.dump(creds, token)
            print(f"Credentials saved to {TOKEN_PATH}")
        except Exception as e:
            print(f"Error saving token to {TOKEN_PATH}: {e}")


    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except HttpError as error:
        print(f'An error occurred building the service: {error}')
        return None
    except Exception as e:
        print(f'An unexpected error occurred building the service: {e}')
        return None


def upload_file(service, file_path, drive_filename):
    """Uploads a file to Google Drive and returns the file ID."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    print(f"Starting upload of '{file_path}' as '{drive_filename}'...")
    file_metadata = {'name': drive_filename}
    media = MediaFileUpload(file_path, mimetype='application/zip', resumable=True)
    try:
        file = service.files().create(body=file_metadata,
                                      media_body=media,
                                      fields='id').execute()
        print(f"File '{drive_filename}' uploaded successfully.")
        return file.get('id')
    except HttpError as error:
        print(f'An HTTP error occurred during upload: {error}')
        # Attempt to parse specific Drive API errors if possible
        error_content = getattr(error, 'content', None)
        if error_content:
            try:
                import json
                error_details = json.loads(error_content.decode('utf-8'))
                print(f"Error details: {error_details}")
            except Exception:
                print(f"Raw error content: {error_content}")
        return None
    except Exception as e:
         print(f'An unexpected error occurred during upload: {e}')
         return None

def set_public_permissions(service, file_id):
    """Sets file permissions to allow anyone with the link to read."""
    try:
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(fileId=file_id, body=permission).execute()
        print("Permissions set to 'anyone with the link can view'.")
        return True
    except HttpError as error:
        print(f'An error occurred setting permissions: {error}')
        return False
    except Exception as e:
         print(f'An unexpected error occurred setting permissions: {e}')
         return False

def get_shareable_link(service, file_id):
    """Gets the shareable web view link for the file."""
    try:
        file = service.files().get(fileId=file_id, fields='webViewLink').execute()
        return file.get('webViewLink')
    except HttpError as error:
        print(f'An error occurred getting the shareable link: {error}')
        return None
    except Exception as e:
         print(f'An unexpected error occurred getting the shareable link: {e}')
         return None

if __name__ == '__main__':
    print("Authenticating with Google Drive...")
    drive_service = authenticate()

    if drive_service:
        print("Authentication successful.")
        file_id = upload_file(drive_service, FILE_TO_UPLOAD, DRIVE_FILENAME)

        if file_id:
            print(f"File ID: {file_id}")
            if set_public_permissions(drive_service, file_id):
                share_link = get_shareable_link(drive_service, file_id)
                if share_link:
                    print("\n--- Success! ---")
                    print(f"Shareable Link: {share_link}")
                else:
                    print("Could not retrieve the shareable link.")
            else:
                print("Failed to set public permissions.")
        else:
            print("File upload failed.")
    else:
        print("Failed to authenticate with Google Drive.")
