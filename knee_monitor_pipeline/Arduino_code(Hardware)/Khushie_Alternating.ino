#include <Wire.h>
#include <WiFi.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include <SFE_BMP180.h>
#include <IOXhop_FirebaseESP32.h>
#include "time.h"

// ==== Wi-Fi & Firebase ====
#define WIFI_SSID ""
#define WIFI_PASSWORD ""
#define FIREBASE_HOST ""
#define FIREBASE_AUTH ""

// ==== Sensors ====
MPU6050 mpu(0x69);   // use address 0x68 if 0x69 fails
SFE_BMP180 bmp;

// ==== Timing ====
const unsigned long MODE_DURATION = 60000;  // 1 minute per mode
unsigned long modeStart = 0;
bool mpuMode = true;  // start with MPU mode

// ==== Time (for readable timestamps) ====
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 19800;  // +5:30 India
const int daylightOffset_sec = 0;

// ==== Setup ====
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("=== MPU6050 + BMP180 Alternating Firebase Logger ===");

  // --- I2C setup ---
  Wire.begin(21, 22);
  Wire.setClock(50000); // safer for multiple I2C devices
  delay(200);

  // --- WiFi ---
  Serial.print("Connecting WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.println(WiFi.localIP());
  Firebase.begin(FIREBASE_HOST, FIREBASE_AUTH);

  // --- NTP time ---
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  // --- Sensors ---
  mpu.initialize();
  if (mpu.testConnection()) Serial.println("MPU6050 connected ✅");
  else Serial.println("MPU6050 connection failed ❌");

  if (bmp.begin()) Serial.println("BMP180 ready ✅");
  else Serial.println("BMP180 failed ❌");

  modeStart = millis();
}

// ==== Helper: Timestamp ====
String getTimestamp() {
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    char buff[30];
    strftime(buff, sizeof(buff), "%Y-%m-%d_%H-%M-%S", &timeinfo);
    return String(buff);
  } else return "no_time";
}

// ==== MPU Mode ====
void runMPUMode() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  Serial.print("[MPU] a/g:\t");
  Serial.print(ax); Serial.print("\t");
  Serial.print(ay); Serial.print("\t");
  Serial.print(az); Serial.print("\t");
  Serial.print(gx); Serial.print("\t");
  Serial.print(gy); Serial.print("\t");
  Serial.println(gz);

  String path = "/mpu_readings/" + getTimestamp();
  Firebase.setFloat(path + "/ax", ax);
  Firebase.setFloat(path + "/ay", ay);
  Firebase.setFloat(path + "/az", az);
  Firebase.setFloat(path + "/gx", gx);
  Firebase.setFloat(path + "/gy", gy);
  Firebase.setFloat(path + "/gz", gz);
}

// ==== BMP Mode (Temperature Only) ====
void runBMPMode() {
  char status;
  double T;

  status = bmp.startTemperature();
  if (status != 0) {
    delay(status);
    bmp.getTemperature(T);

    Serial.print("[BMP] Temperature: ");
    Serial.print(T);
    Serial.println(" °C");

    String path = "/bmp_readings/" + getTimestamp();
    Firebase.setFloat(path + "/temperature", T);
  }
}

// ==== Main Loop ====
void loop() {
  if (millis() - modeStart < MODE_DURATION) {
    // Still in current mode
    if (mpuMode) runMPUMode();
    else runBMPMode();
  } else {
    // Switch modes after each minute
    mpuMode = !mpuMode;
    modeStart = millis();
    Serial.println();
    Serial.println("=== Switching mode ===");
    Serial.println(mpuMode ? "MPU6050 active" : "BMP180 active");
    Serial.println();
  }

  delay(1000);  // one reading per second
}

 
