METERS_TO_INCHES_RATIO = 39.3701
METERS_TO_FEET_RATIO = 3.28
METERS_TO_MILE_RATIO = 0.00062137

meters = input("Please, enter number of meters you would like to convert to Imperial measure system: ")

if meters.isnumeric():
    meters = int(meters)
    inches = round((meters * METERS_TO_INCHES_RATIO), 2)
    print("It is {finches} inch(es)".format(finches = inches))

    feet = round((meters * METERS_TO_FEET_RATIO), 2)
    print("It is {ffeet} foot(s)".format(ffeet=feet))

    mile = round((meters * METERS_TO_MILE_RATIO), 2)
    print("It is {fmile} mile(s)".format(fmile=mile))
else:
    print('Please enter a numeric value.')