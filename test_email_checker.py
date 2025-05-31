import subprocess
import os

def test_email_checker(email_file, description):
    print(f"\n{'-'*50}")
    print(f"Test: {description}")
    print(f"File: {email_file}")
    print(f"{'-'*50}")
    
    try:
        with open(email_file, 'r', encoding='utf-8') as f:
            email_content = f.read()
        
        result = subprocess.run(
            ['python', 'email_spam_checker.py'],
            input=email_content,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            output = result.stdout
            print("Çıktı:")
            print(output)
            
            # check if spam is detected
            lines = output.split('\n')
            for line in lines:
                if line.startswith('Subject:'):
                    if '---SPAM---' in line:
                        print("🔴 SPAM detected!")
                    else:
                        print("✅ Normal email")
                    break
        else:
            print(f"Hata: {result.stderr}")
            
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    print("Email Spam Checker Test")
    print("="*50)
    
    # check test files
    test_files = [
        ("test_email_ham.txt", "Normal Email Test"),
        ("test_email_spam.txt", "Spam Email Test")
    ]
    
    for email_file, description in test_files:
        if os.path.exists(email_file):
            test_email_checker(email_file, description)
        else:
            print(f"❌ {email_file} file not found")
    
    print(f"\n{'='*50}")
    print("Test completed!") 