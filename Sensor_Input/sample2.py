import serial
import asyncio

# Configure serial connection
ser = serial.Serial('COM6', 9600)  # Replace 'COM4' with your port and 9600 with your baud rate

async def read_serial():
    with open('sensor_data2.txt', 'w') as file:  # Open the file in write mode
        while True:
            try:
                data = ser.readline().decode('utf-8').strip()  # Read and decode the serial data
                print(data)  # Print the received data to the console
                
                # Write the data to the file
                file.write(data + '\n')
                file.flush()  # Ensure the data is written to the file immediately
            except Exception as e:
                print(f"Error: {e}")

# Run the async function
asyncio.run(read_serial())
