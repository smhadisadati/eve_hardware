#include <Arduino.h>
#include <Encoder.h>

Encoder linearEncoder(24, 2);
Encoder rotationEncoder(5, 4);

long initLinPos_wire ;
long initRotPos_wire ;
long initLinPos_mcath ;
long initRotPos_mcath ;
long initLinPos_cath ;
long initRotPos_cath ;

long resetLinPos_wire = 0 ;
long resetRotPos_wire = 0 ;
long resetLinPos_mcath = 0 ;
long resetRotPos_mcath = 0 ;
long resetLinPos_cath = 0 ;
long resetRotPos_cath = 0 ;

long oldLinPos_wire = 0 ;
long oldRotPos_wire = 0 ;
long oldLinPos_mcath = 0 ;
long oldRotPos_mcath = 0 ;
long oldLinPos_cath = 0 ;
long oldRotPos_cath = 0 ;

long linMov_wire;
long rotMov_wire;
long linMov_mcath;
long rotMov_mcath;
long linMov_cath;
long rotMov_cath;

long linVel_wire ;
long rotVel_wire ;
long linVel_mcath ;
long rotVel_mcath ;
long linVel_cath ;
long rotVel_cath ;

const int green_LEDPin = 28; // broken LED or bad wiring
const int yellow_LEDPin = 26;
const int white_LEDPin = 32;
bool green_LEDState = LOW; // bLED state
bool yellow_LEDState = LOW;
bool white_LEDState = LOW;

const int green_buttonPin = 7;
const int yellow_buttonPin = 6;
const int white_buttonPin = 22; // Needs soldering
const int black_buttonPin = 30;

// variables will change:
int buttonState_black = 0;
int buttonState_yellow = 0;
int buttonState_green = 0;
int buttonState_white = 0;

// PWM
// Lin on cahnnel A
int directionPin_lin = 12;
int pwmPin_lin = 3;
int brakePin_lin = 9;

// Rot on channel B
int directionPin_rot = 13;
int pwmPin_rot = 11;
int brakePin_rot = 8;

//boolean to switch direction
int pwm_lin = 0;
int pwm_rot = 0;

int inChar ;
String outString ;
String inString = "";

void setup() {
  Serial.begin(115200);

  //Setup Channel A
  pinMode(12, OUTPUT); //Initiates Motor Channel A pin
  pinMode(9, OUTPUT); //Initiates Brake Channel A pin

  //Setup Channel B
  pinMode(13, OUTPUT); //Initiates Motor Channel B pin
  pinMode(8, OUTPUT); //Initiates Brake Channel B pin

  // initialize the LED pin as an output:
  pinMode(white_LEDPin, OUTPUT);
  pinMode(yellow_LEDPin, OUTPUT);
  pinMode(green_LEDPin, OUTPUT);

  // initialize the pushbutton pin as an input:
  pinMode(white_buttonPin, INPUT);
  pinMode(yellow_buttonPin, INPUT);
  pinMode(green_buttonPin, INPUT);

  // PWM
  pinMode(directionPin_lin, OUTPUT);
  pinMode(pwmPin_lin, OUTPUT);
  pinMode(brakePin_lin, OUTPUT);
  
  pinMode(directionPin_rot, OUTPUT);
  pinMode(pwmPin_rot, OUTPUT);
  pinMode(brakePin_rot, OUTPUT);
  
  //release breaks
  digitalWrite(brakePin_lin, LOW);
  digitalWrite(brakePin_rot, LOW);

  // home encoder
  initLinPos_wire = linearEncoder.read();
  initRotPos_wire = rotationEncoder.read();
  initLinPos_mcath = linearEncoder.read();
  initRotPos_mcath = rotationEncoder.read();
  initLinPos_cath = linearEncoder.read();
  initRotPos_cath = rotationEncoder.read();

}

void loop() {
  // encoder read
  long newLinPos = linearEncoder.read();
  long newRotPos = rotationEncoder.read();

  // read the state of the pushbutton value:
  buttonState_black = digitalRead(black_buttonPin); // all three
  buttonState_yellow = digitalRead(yellow_buttonPin); // guidewire
  buttonState_green = digitalRead(green_buttonPin); // micfo catheter
  buttonState_white = digitalRead(white_buttonPin); // bocatheterth

  if (buttonState_yellow == HIGH || buttonState_green == HIGH || buttonState_white == HIGH || buttonState_black == HIGH ) { // reset state
    // LED state update
    if (buttonState_yellow == HIGH ){
      yellow_LEDState = HIGH ;
      green_LEDState = LOW ;
      white_LEDState = LOW ;
      initLinPos_wire = linearEncoder.read();
      initRotPos_wire = rotationEncoder.read();
    }
    if (buttonState_green == HIGH ){
      yellow_LEDState = LOW ;
      green_LEDState = HIGH ;
      white_LEDState = LOW ;
      initLinPos_mcath = linearEncoder.read();
      initRotPos_mcath = rotationEncoder.read();
    }
    if (buttonState_white == HIGH ){
      yellow_LEDState = LOW ;
      green_LEDState = LOW ;
      white_LEDState = HIGH ;
      initLinPos_cath = linearEncoder.read();
      initRotPos_cath = rotationEncoder.read();
    }
    if ( buttonState_black == HIGH ){
      yellow_LEDState = HIGH ;
      green_LEDState = HIGH ;
      white_LEDState = HIGH ;
      initLinPos_wire = linearEncoder.read();
      initRotPos_wire = rotationEncoder.read();
      initLinPos_mcath = linearEncoder.read();
      initRotPos_mcath = rotationEncoder.read();
      initLinPos_cath = linearEncoder.read();
      initRotPos_cath = rotationEncoder.read();
    }
    
    resetLinPos_wire = oldLinPos_wire ;
    resetRotPos_wire = oldRotPos_wire ;
    resetLinPos_mcath = oldLinPos_mcath ;
    resetRotPos_mcath = oldRotPos_mcath ;
    resetLinPos_cath = oldLinPos_cath ;
    resetRotPos_cath = oldRotPos_cath ;
    
    linMov_wire = resetLinPos_wire ;
    rotMov_wire = resetRotPos_wire ;
    linMov_mcath = resetLinPos_mcath ;
    rotMov_mcath = resetRotPos_mcath ;
    linMov_cath = resetLinPos_cath ;
    rotMov_cath = resetRotPos_cath ;
    
    linVel_wire = 0 ;
    rotVel_wire = 0 ;
    linVel_mcath = 0 ;
    rotVel_mcath = 0 ;
    linVel_cath = 0 ;
    rotVel_cath = 0;
    
  }
  else { // drive state
    if ( yellow_LEDState == HIGH ) {
      linMov_wire = newLinPos - initLinPos_wire + resetLinPos_wire;
      rotMov_wire = newRotPos - initRotPos_wire + resetRotPos_wire;
    }
    if ( green_LEDState == HIGH ) {
      linMov_mcath = newLinPos - initLinPos_mcath + resetLinPos_mcath;
      rotMov_mcath = newRotPos - initRotPos_mcath + resetRotPos_mcath;
    }
    if ( white_LEDState == HIGH ) {
      linMov_cath = newLinPos - initLinPos_cath + resetLinPos_cath;
      rotMov_cath = newRotPos - initRotPos_cath + resetRotPos_cath;
    }
    if ( yellow_LEDState == HIGH &&  green_LEDState == HIGH &&  white_LEDState == HIGH ) {
      linMov_wire = newLinPos - initLinPos_wire + resetLinPos_wire;
      rotMov_wire = newRotPos - initRotPos_wire + resetRotPos_wire;
      linMov_mcath = newLinPos - initLinPos_mcath + resetLinPos_mcath;
      rotMov_mcath = newRotPos - initRotPos_mcath + resetRotPos_mcath;
      linMov_cath = newLinPos - initLinPos_cath + resetLinPos_cath;
      rotMov_cath = newRotPos - initRotPos_cath + resetRotPos_cath;
    }

    linVel_wire = linMov_wire - oldLinPos_wire ;
    rotVel_wire = rotMov_wire - oldRotPos_wire ;
    linVel_mcath = linMov_mcath - oldLinPos_mcath ;
    rotVel_mcath = rotMov_mcath - oldRotPos_mcath ;
    linVel_cath = linMov_cath - oldLinPos_cath ;
    rotVel_cath = rotMov_cath - oldRotPos_cath ;

    oldLinPos_wire = linMov_wire ;
    oldRotPos_wire = rotMov_wire ;
    oldLinPos_mcath = linMov_mcath ;
    oldRotPos_mcath = rotMov_mcath ;
    oldLinPos_cath = linMov_cath ;
    oldRotPos_cath = rotMov_cath ;
  }

  // set LED states
  digitalWrite(yellow_LEDPin, yellow_LEDState);
  digitalWrite(green_LEDPin, green_LEDState);
  digitalWrite(white_LEDPin, white_LEDState);
  

  // read serial 
  while( Serial.available() > 0 ) {
    inChar = Serial.read();
    if ( isDigit(inChar) ) {
      // convert the incoming byte to a char and add it to the string:
      inString += (char)inChar;
    }
//    else {
//      break ;
//    }
    // if you get a newline, print the string, then the string's value:
    if (inChar == ',') {
      pwm_lin = inString.toInt() ;
      inString = "";
      continue;
    }
    if ( inChar == '\n' ) {
      pwm_rot = inString.toInt() ;
      inString = "";
      break;
    }
  } 
  // Serial.println(String(pwm_lin) + ',' + String(pwm_rot));

  // PWM part: resistance direction
  if (pwm_lin <= 0 ){
    digitalWrite(directionPin_lin, LOW);
  }
  else{
    digitalWrite(directionPin_lin, HIGH);
  }

  if (pwm_rot <= 0 ){
    digitalWrite(directionPin_rot, LOW);
  }
  else{
    digitalWrite(directionPin_rot, HIGH);
  }

  //set work duty for the motor
  analogWrite(pwmPin_lin, abs(pwm_lin));
  analogWrite(pwmPin_rot, abs(pwm_rot));

  // write serial
  outString = String(linMov_wire) + ',' + String(rotMov_wire) + ',' + String(linMov_mcath) + ',' + String(rotMov_mcath) + ',' + String(linMov_cath) + ',' + String(rotMov_cath) + ',' + String(linVel_wire) + ',' + String(rotVel_wire) + ',' + String(linVel_mcath) + ',' + String(rotVel_mcath) + ',' + String(linVel_cath) + ',' + String(rotVel_cath) + ',' + String(pwm_lin) + ',' + String(pwm_rot);
  Serial.println(outString);
}
