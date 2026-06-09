import os
import sys

# Load dotenv to configure twilio environment mock
from dotenv import load_dotenv
load_dotenv()

from bot import handle_message

def run_tests():
    print("=== RUNNING LOCAL WEBHOOK MULTILINGUAL DISPATCHER TESTS ===\n")
    
    # Test 1: English News Claim
    print("[TEST 1] English News Claim Analysis")
    claim = "The central bank is shutting down all digital ATMs tomorrow due to security issues."
    resp = handle_message(claim)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")
    
    # Test 2: Hindi News Claim
    print("[TEST 2] Hindi News Claim Analysis")
    claim_hi = "सरकार सभी नागरिकों को मुफ्त में 5G मोबाइल बांट रही है।"
    resp = handle_message(claim_hi)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 3: Telugu News Claim
    print("[TEST 3] Telugu News Claim Analysis")
    claim_te = "ఈ కొత్త వ్యాక్సిన్ వేసుకోవడం వల్ల ప్రజలు అనారోగ్యానికి గురవుతున్నారు."
    resp = handle_message(claim_te)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 4: Short Claim in Hindi (< 6 words)
    print("[TEST 4] Short Claim Guardrail in Hindi")
    short_hi = "मोबाइल मुफ्त में"
    resp = handle_message(short_hi)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 5: Help Welcome message check (Empty body)
    print("[TEST 5] Help/Welcome Message in Hindi")
    resp = handle_message("मदद") # Triggers Hindi detection, showing Hindi welcome guide
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

    # Test 6: AI Text Detector Command (/ai)
    print("[TEST 6] AI Text Detector Command (/ai)")
    ai_cmd = "/ai It is crucial to leverage comprehensive strategies that facilitate robust outcomes. Furthermore, this ensures streamlined implementation."
    resp = handle_message(ai_cmd)
    print("Response:")
    print(resp)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    run_tests()
