from simplegmail import Gmail
from Detector import predict_phishing
import time

gmail = Gmail()
labels = gmail.list_labels()

checking_label = list(filter(lambda x: x.name == 'Wird geprüft', labels))[0]
valid_label = list(filter(lambda x: x.name == 'Passt', labels))[0]
phishing_label = list(filter(lambda x: x.name == 'Phishing', labels))[0]

def save_emails_to_files():

    messages = gmail.get_unread_inbox(labels=[checking_label])

    for message in messages:
        check_phishing(message.plain, message)

def check_phishing(plain, message):
  is_phishing = predict_phishing(plain)
  print(is_phishing)
  if (is_phishing):
    message.modify_labels(to_add=phishing_label, to_remove=checking_label)
  else:
    message.modify_labels(to_add=valid_label, to_remove=checking_label)


def main():
    while True:
        save_emails_to_files()
        time.sleep(1)

if __name__ == "__main__":
    main()