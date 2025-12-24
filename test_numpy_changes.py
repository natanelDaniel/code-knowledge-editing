import numpy as np
import os
import subprocess
import io
import sys
import torch

def run_code_snippet(code_str):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_stdout = io.StringIO()
    redirected_stderr = io.StringIO()
    sys.stdout = redirected_stdout
    sys.stderr = redirected_stderr
    
    returncode = 0
    try:
        exec(code_str, globals())
    except Exception as e:
        sys.stderr.write(str(e))
        returncode = 1 # Indicate an error occurred
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return returncode, redirected_stdout.getvalue(), redirected_stderr.getvalue()

if __name__ == "__main__":
    all_code_examples = [
                {
            "name": "31. torch.range() Deprecation (Pre 0.4)",
            "old_code": """
import torch
# torch.range() is deprecated in favor of torch.arange()
# In older versions, this might have worked or given a different warning.
# Now, it will fail because arguments are required.
result = torch.range(1, 5)
print(result)
""",
            "new_code": """
import torch
# Use torch.arange() instead of torch.range().
# Note: torch.range(start, end) is inclusive of end, while torch.arange(start, end) is exclusive of end.
result = torch.arange(1, 6)
print(result)
""",
            "expected_old_error": "DeprecationWarning" # For older versions, it might be a DeprecationWarning, then TypeError/RuntimeError.
        },
        {
            "name": "32. torch.chain_matmul() Deprecation (Moved to linalg)",
            "old_code": """
import torch

a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.randn(4, 5)
# torch.chain_matmul() is deprecated in favor of torch.linalg.multi_dot()
# This will likely raise a RuntimeError or AttributeError in newer versions.
result = torch.chain_matmul(a, b, c)
print(result)
""",
            "new_code": """
import torch

a = torch.randn(2, 3)
b = torch.randn(3, 4)
c = torch.randn(4, 5)
# Use torch.linalg.multi_dot() for chained matrix multiplication.
result = torch.linalg.multi_dot([a, b, c])
print(result)
""",
            "expected_old_error": "AttributeError" # Expected to be removed from top-level torch.
        },
        {
            "name": "33. torch.set_deterministic() Deprecation (Replaced by use_deterministic_algorithms)",
            "old_code": """
import torch

# torch.set_deterministic() is deprecated.
# This will likely raise an AttributeError in newer versions.
torch.set_deterministic(True)
print("Deterministic mode set (old method).")
""",
            "new_code": """
import torch

# Use torch.use_deterministic_algorithms() and related functions.
torch.use_deterministic_algorithms(True)
print("Deterministic mode set (new method).")
""",
            "expected_old_error": "AttributeError" # Expected to be removed from top-level torch.
        },
        {
            "name": "1. numpy.take_along_axis default axis (Pre 2.3.0)",
            "old_code": """
import numpy as np
arr = np.array([[10, 30, 20], [60, 50, 40]])
indices = np.array([[0, 2, 1], [2, 0, 1]])
# בגרסאות ישנות (לפני 2.3), זה אמור להעלות שגיאה כי axis נדרש.
result = np.take_along_axis(arr, indices)
print(result)
""",
            "new_code": """
import numpy as np
arr = np.array([[10, 30, 20], [60, 50, 40]])
indices = np.array([[0, 2, 1], [2, 0, 1]])
# בגרסאות חדשות (2.3+), זה עובד עם ברירת מחדל axis=-1.
result = np.take_along_axis(arr, indices, axis=-1)
print(result)
""",
            "expected_old_error": "TypeError"
        },
        {
            "name": "2. np.concatenate with empty list and axis (Pre 2.3.0)",
            "old_code": """
import numpy as np
list_of_arrays = []
# בגרסאות ישנות, זה החזיר מערך ריק ללא שגיאה.
result = np.concatenate(list_of_arrays, axis=0)
print(result)
""",
            "new_code": """
import numpy as np
list_of_arrays = []
# בגרסאות חדשות (2.3+), np.concatenate מעלה ValueError אם axis מצוין.
if not list_of_arrays:
    result = np.array([])
else:
    result = np.concatenate(list_of_arrays, axis=0)
print(result)
""",
            "expected_old_error": "ValueError"
        },
        {
            "name": "3. np.array_equal comparison to scalar (Logical Failure Pre 2.3.0)",
            "old_code": """
import numpy as np
arr = np.array([1, 1, 1])
# בגרסאות ישנות, זה החזיר True (כשל לוגי ב-2.3+).
if np.array_equal(arr, 1):
    print("Logic executed based on True.")
else:
    print("Logic executed based on False.")
""",
            "new_code": """
import numpy as np
arr = np.array([1, 1, 1])
# בגרסאות חדשות, יש להשתמש ב-np.all(arr == 1) להשוואה אלמנט-אלמנט.
if np.all(arr == 1):
    print("Logic executed based on True.")
else:
    print("Logic executed based on False.")
""",
            "expected_old_error": "False" # כשל לוגי, לא שגיאת ריצה מפורשת.
        },
        {
            "name": "4. np.histogram with non-finite bins (Pre 1.21.0)",
            "old_code": """
import numpy as np
arr = np.array([1, 2, 3])
bins = np.array([0, 1, np.inf]) # שימוש ב-inf
# בגרסאות ישנות, זה עשוי היה לעבוד או להחזיר תוצאה לא עקבית.
hist, edges = np.histogram(arr, bins=bins)
print(hist)
""",
            "new_code": """
import numpy as np
arr = np.array([1, 2, 3])
# בגרסאות חדשות (1.21.0+), יש להבטיח שכל ה-bins סופיים.
# נשתמש ב-bins סופיים עבור הקוד החדש כדי להצליח.
hist, edges = np.histogram(arr, bins=np.array([0, 1, 4]))
print(hist)
""",
            "expected_old_error": "ValueError" # גרסאות 1.21+ מעלות ValueError כאשר bins אינם סופיים.
        },
        {
            "name": "5. np.clip with incompatible 'out' dtype (Pre 1.25.0)",
            "old_code": """
import numpy as np
a = np.array([1.1, 2.8, 3.3])
out_arr = np.empty_like(a, dtype=np.int32)
# בגרסאות ישנות, clip היה יותר סלחן וביצע את ה-casting.
np.clip(a, 1, 3, out=out_arr)
print(out_arr)
""",
            "new_code": """
import numpy as np
a = np.array([1.1, 2.8, 3.3])
# בגרסאות חדשות (1.25.0+), יש לבצע את ה-casting באופן מפורש.
result = np.clip(a, 1, 3).astype(np.int32)
print(result)
""",
            "expected_old_error": "TypeError" # גרסאות 1.25+ מעלות שגיאה כשמבצעים clip ל-out עם dtype שונה (ולא תואם).
        },
        {
            "name": "6. Removal of np.float/np.int aliases (Pre 2.0.0)",
            "old_code": """
import numpy as np
# בגרסאות ישנות, ניתן היה להשתמש בכינוי זה.
dt = np.float
print(dt)
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (2.0.0+), יש להשתמש בסוג המובנה של Python.
dt = float
print(dt)
""",
            "expected_old_error": "AttributeError" # np.float הוסר וגרם לשגיאה.
        },
        # --- דוגמאות חדשות ---
        {
            "name": "7. np.random.rand with list as size (Pre 2.0.0)",
            "old_code": """
import numpy as np
size_list = [2, 3]
# בגרסאות ישנות, העברת רשימה כארגומנט יחיד עבור size עבדה.
result = np.random.rand(size_list)
print(result.shape)
""",
            "new_code": """
import numpy as np
size_tuple = (2, 3)
# בגרסאות חדשות (2.0.0+), size חייב להיות tuple או מספרים נפרדים.
result = np.random.rand(*size_tuple)
print(result.shape)
""",
            "expected_old_error": "TypeError" # 2.0.0+ דורש tuple.
        },
        {
            "name": "8. np.testing.assert_array_equal bytes/str (Pre 1.20.0)",
            "old_code": """
import numpy as np
# בגרסאות ישנות, היתה המרה שקטה בין bytes ו-str בהשוואה.
np.testing.assert_array_equal(np.array([b'a']), np.array(['a']))
print("Assertion passed.")
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (1.20.0+), נדרשת השוואה בין סוגים זהים.
np.testing.assert_array_equal(np.array([b'a']), np.array([b'a']))
print("Assertion passed.")
""",
            "expected_old_error": "TypeError" # 1.20.0+ מסרב להמיר ומעלה שגיאה.
        },
        {
            "name": "9. np.broadcast_arrays with non-array object (Pre 1.22.0)",
            "old_code": """
import numpy as np
class Custom:
    pass
# בגרסאות ישנות, ניתן היה להעביר אובייקטים שאינם מערכים.
result = np.broadcast_arrays(Custom(), 2)
print("Broadcast succeeded.")
""",
            "new_code": """
import numpy as np
class Custom:
    pass
# בגרסאות חדשות (1.22.0+), יש להמיר במפורש מערכים שאינם סקלרים.
result = np.broadcast_arrays(np.array(Custom(), dtype=object), np.array(2))
print("Broadcast succeeded.")
""",
            "expected_old_error": "TypeError" # 1.22.0+ דורש קלט תואם מערך.
        },
        {
            "name": "10. np.linalg.solve dtype mismatch (Pre 1.21.0)",
            "old_code": """
import numpy as np
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([1, 2], dtype=np.float64) # dtype שונה
# בגרסאות ישנות, עשוי היה להצליח עם אזהרה.
np.linalg.solve(a, b)
print("Solve succeeded.")
""",
            "new_code": """
import numpy as np
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([1, 2], dtype=np.float64).astype(np.float32) # המרה מפורשת
# בגרסאות חדשות (1.21.0+), נדרשת התאמת dtype.
np.linalg.solve(a, b)
print("Solve succeeded.")
""",
            "expected_old_error": "TypeError" # 1.21.0+ מעלה שגיאה על חוסר התאמה ב-dtype ב-linalg.
        },
        {
            "name": "11. np.percentile empty array (Pre 1.23.0)",
            "old_code": """
import numpy as np
arr = np.array([])
# בגרסאות ישנות, קריאה זו הצליחה והחזירה NaN או עוררה אזהרה.
result = np.percentile(arr, 50)
print(result)
""",
            "new_code": """
import numpy as np
arr = np.array([])
# בגרסאות חדשות (1.23.0+), זה מעלה ValueError אם המערך ריק.
if arr.size == 0:
    result = np.nan
else:
    result = np.percentile(arr, 50)
print(result)
""",
            "expected_old_error": "ValueError" # 1.23.0+ מעלה שגיאה על מערך ריק.
        },
        {
            "name": "12. np.array subclass construction (Pre 1.25.0)",
            "old_code": """
import numpy as np
class MySubClass(np.ndarray):
    pass
# בגרסאות ישנות, ניתן היה ליצור תת-מחלקה כך.
result = np.array(10, dtype=MySubClass)
print(result)
""",
            "new_code": """
import numpy as np
class MySubClass(np.ndarray):
    pass
# בגרסאות חדשות (1.25.0+), יש לבנות את תת-המחלקה ישירות.
result = MySubClass(10)
print(result)
""",
            "expected_old_error": "TypeError" # 1.25.0+ מעלה שגיאה כשמנסים לבנות תת-מחלקה דרך dtype.
        },
        {
            "name": "13. np.fft.fft non-contiguous input (Pre 1.24.0)",
            "old_code": """
import numpy as np
a = np.arange(10)[::2] # מערך לא רציף
# בגרסאות ישנות, פונקציות FFT עבדו לעיתים עם קלט לא רציף.
np.fft.fft(a)
print("FFT succeeded.")
""",
            "new_code": """
import numpy as np
a = np.arange(10)[::2]
# בגרסאות חדשות (1.24.0+), קלט FFT חייב להיות רציף, לכן נדרש copy.
np.fft.fft(a.copy())
print("FFT succeeded.")
""",
            "expected_old_error": "ValueError" # 1.24.0+ מעלה שגיאה על קלט לא רציף.
        },
        {
            "name": "14. np.dtype string aliases (Pre 2.0.0)",
            "old_code": """
import numpy as np
# בגרסאות ישנות, aliases אלו עבדו (למשל, 'f' עבור 'f4').
dt = np.dtype('f')
print(dt)
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (2.0.0+), יש להשתמש בשמות מפורטים.
dt = np.dtype('f4')
print(dt)
""",
            "expected_old_error": "TypeError" # 2.0.0+ מסיר aliases קצרים מסוימים ב-dtype.
        },
        {
            "name": "15. np.array with shape misalignment (Pre 1.23.0)",
            "old_code": """
import numpy as np
# יצירת מערך סקלרי עם shape שאינו תואם את הקלט.
np.array(1, dtype=[('f', 'i4')], shape=(2,))
print("Array created.")
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (1.23.0+), ה-shape חייב להתאים.
np.array([(1,), (2,)], dtype=[('f', 'i4')])
print("Array created.")
""",
            "expected_old_error": "ValueError" # 1.23.0+ מעלה שגיאה על חוסר התאמה בין קלט סקלרי ל-shape.
        },
        {
            "name": "16. np.result_type complex promotion (Pre 2.0.0)",
            "old_code": """
import numpy as np
# בגרסאות ישנות, קידום סוגים מסוימים ל-object היה אפשרי.
common = np.result_type(np.int8, object)
print(common)
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (2.0.0+), קידום ל-object הוסר.
common = np.result_type(np.int8, float) # יש להשתמש בסוג קונקרטי.
print(common)
""",
            "expected_old_error": "TypeError" # 2.0.0+ מעלה שגיאה בעת ניסיון קידום אוטומטי ל-object.
        },
        {
            "name": "17. np.pad with mode='constant' and stat_length (Pre 1.25.0)",
            "old_code": """
import numpy as np
arr = np.array([1, 2, 3])
# בגרסאות ישנות, ציון stat_length עם mode='constant' היה מעלה שגיאה.
result = np.pad(arr, (1, 1), mode='constant', constant_values=0, stat_length=((0,0)))
print(result)
""",
            "new_code": """
import numpy as np
arr = np.array([1, 2, 3])
# בגרסאות חדשות (1.25.0+), stat_length מתעלם מ-mode='constant'.
result = np.pad(arr, (1, 1), mode='constant', constant_values=0)
print(result)
""",
            "expected_old_error": "ValueError" # 1.25.0+ מעלה שגיאה.
        },
        {
            "name": "18. np.average with weights and returned=True and axis=None (Pre 2.0.0)",
            "old_code": """
import numpy as np
data = np.array([1, 2, 3])
weights = np.array([1, 1, 2])
# בגרסאות ישנות, returned_weights היה מערך (אפילו אם סקלרי), מה שיכול לגרום לשגיאות בבדיקת צורות.
avg, sum_weights = np.average(data, weights=weights, returned=True)
print(sum_weights.shape)
""",
            "new_code": """
import numpy as np
data = np.array([1, 2, 3])
weights = np.array([1, 1, 2])
# בגרסאות חדשות (2.0.0+), sum_weights הוא תמיד סקלר כאשר axis=None.
avg, sum_weights = np.average(data, weights=weights, returned=True)
print(np.asarray(sum_weights).shape)
""",
            "expected_old_error": "TypeError" # תלוי בשימוש, אך שינוי צורה עלול להוביל לשגיאות מסוג זה.
        },
        {
            "name": "19. np.load with allow_pickle=False and pickled data (Pre 1.25.0)",
            "old_code": """
import numpy as np
import pickle
import os

# יצירת קובץ npz עם נתונים מפוקסלים
data_to_save = {'a': np.array([1, 2]), 'b': pickle.dumps([4, 5])}
np.savez('temp_pickled_data.npz', **data_to_save)

# בגרסאות ישנות, זה עשוי היה לטעון ללא שגיאה גם עם allow_pickle=False עבור חלק מהמקרים.
try:
    loaded_data = np.load('temp_pickled_data.npz', allow_pickle=False)
    print("Old code loaded data (might be unsafe):")
    # Accessing 'b' might still work, which is the vulnerability
    print(loaded_data['b'])
except Exception as e:
    print(f"Old code failed as expected: {e}")
finally:
    os.remove('temp_pickled_data.npz')
""",
            "new_code": """
import numpy as np
import pickle
import os

# יצירת קובץ npz עם נתונים מפוקסלים
data_to_save = {'a': np.array([1, 2]), 'b': pickle.dumps([4, 5])}
np.savez('temp_pickled_data.npz', **data_to_save)

# בגרסאות חדשות (1.25.0+), זה מעלה ValueError אם הוא מזהה נתונים מפוקסלים כש-allow_pickle=False.
try:
    loaded_data = np.load('temp_pickled_data.npz', allow_pickle=False)
    print("New code loaded data:")
except ValueError as e:
    print(f"New code failed as expected with ValueError: {e}")
finally:
    os.remove('temp_pickled_data.npz')
""",
            "expected_old_error": "ValueError" # 1.25.0+ מעלה שגיאה.
        },
        {
            "name": "20. Truth value of empty array (Pre 2.2.0)",
            "old_code": """
import numpy as np
arr = np.array([])
# בגרסאות ישנות (לפני 2.2.0), הערכת ערך האמת של מערך ריק עשויה היתה להצליח.
# היא החזירה False או עוררה אזהרה.
if arr:
    print("Array is truthy.")
else:
    print("Array is falsy.")
""",
            "new_code": """
import numpy as np
arr = np.array([])
# בגרסאות חדשות (2.2.0+), הערכת ערך האמת של מערך ריק מעלה ValueError.
# יש להשתמש ב-arr.size > 0 כדי לבדוק אם המערך אינו ריק.
if arr.size > 0:
    print("Array is truthy.")
else:
    print("Array is falsy.")
""",
            "expected_old_error": "ValueError" # 2.2.0+ מעלה שגיאה.
        },
        {
            "name": "21. np.nonzero with scalar or 0D array (Pre 2.1.0)",
            "old_code": """
import numpy as np
arr_scalar = np.array(5)
arr_0d = np.array(True)
# בגרסאות ישנות (לפני 2.1.0), ניתן היה לקרוא ל-nonzero על סקלרים או מערכים 0D.
result_scalar = np.nonzero(arr_scalar)
result_0d = arr_0d.nonzero()
print(result_scalar)
print(result_0d)
""",
            "new_code": """
import numpy as np
arr_scalar = np.array(5)
arr_0d = np.array(True)
# בגרסאות חדשות (2.1.0+), קריאה ל-nonzero על סקלרים או מערכים 0D מעלה ValueError.
# יש להשתמש בבדיקות מפורשות כמו arr_scalar.item() או arr_0d.item() == True.
try:
    result_scalar = np.nonzero(arr_scalar)
except ValueError as e:
    print(f"New code caught expected error for scalar: {e}")
try:
    result_0d = arr_0d.nonzero()
except ValueError as e:
    print(f"New code caught expected error for 0D array: {e}")
""",
            "expected_old_error": "ValueError" # 2.1.0+ מעלה שגיאה.
        },
        {
            "name": "22. np.array with copy=False on non-writeable object (Pre 2.0.0)",
            "old_code": """
import numpy as np
arr = np.arange(5)
arr.flags.writeable = False
# בגרסאות ישנות (לפני 2.0.0), יצירת מערך עם copy=False מאובייקט לא-כתוב עשויה הייתה לעבוד.
new_arr = np.array(arr, copy=False)
print(new_arr)
""",
            "new_code": """
import numpy as np
arr = np.arange(5)
arr.flags.writeable = False
# בגרסאות חדשות (2.0.0+), ניסיון ליצור מערך עם copy=False מאובייקט לא-כתוב מעלה ValueError.
# יש להשתמש ב-copy=True או להפוך את המערך לכתוב. (או arr.copy() אם מעוניינים בעותק)
try:
    new_arr = np.array(arr, copy=False)
except ValueError as e:
    print(f"New code caught expected error: {e}")
new_arr_safe = np.array(arr.copy(), copy=False) # יצירת עותק כתוב לפני העברה
print(new_arr_safe)
""",
            "expected_old_error": "ValueError" # 2.0.0+ מעלה שגיאה.
        },
        {
            "name": "23. np.testing.assert_array_equal with strict=True (Pre 1.24.0)",
            "old_code": """
import numpy as np
# בגרסאות ישנות (לפני 1.24.0), השוואת מערכים עם סקלר או dtype שונה עשויה הייתה לעבוד
# או להחזיר תוצאה שונה ללא שגיאה, אם לא צוין אחרת.
# נשתמש בדוגמה של dtype שונה כדי להראות את השינוי.
a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([1, 2, 3], dtype=np.int32)
# בעבר, השוואה כזו הייתה עוברת ללא שגיאה (אולי עם המרה שקטה).
np.testing.assert_array_equal(a, b)
print("Assertion passed (old behavior).")
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (1.24.0+), strict=True בודק גם התאמת dtype ומעלה AssertionError.
a = np.array([1, 2, 3], dtype=np.float32)
b = np.array([1, 2, 3], dtype=np.int32)
# כדי להבטיח את אותה פונקציונליות או לטפל בשגיאה, יש להמיר את ה-dtype במפורש.
try:
    np.testing.assert_array_equal(a, b, strict=True)
except AssertionError as e:
    print(f"New code caught expected AssertionError with strict=True: {e}")
# לדוגמה, אם רוצים להשוות ערכים בלבד, ניתן לבצע המרה:
np.testing.assert_array_equal(a.astype(b.dtype), b)
print("Assertion passed (new behavior with type casting).")
""",
            "expected_old_error": "AssertionError" # 1.24.0+ מעלה שגיאה כשמבצעים השוואה עם strict=True ו-dtype שונה.
        },
        {
            "name": "24. np.loadtxt parsing float into integer dtype (Pre 1.23.0)",
            "old_code": """
import numpy as np
import io

data_str = "1.0\n2.5\n3.0"
data_file = io.StringIO(data_str)

# בגרסאות ישנות (לפני 1.23.0), טעינת ערכים עשרוניים ל-dtype שלם עשויה הייתה להצליח עם קיטום.
result = np.loadtxt(data_file, dtype=int)
print(result)
""",
            "new_code": """
import numpy as np
import io

data_str = "1.0\n2.5\n3.0"
data_file = io.StringIO(data_str)

# בגרסאות חדשות (1.23.0+), ניסיון לטעון ערכים עשרוניים ל-dtype שלם מעלה ValueError.
# יש לטעון ל-float ואז להמיר במפורש.
try:
    result = np.loadtxt(data_file, dtype=int)
except ValueError as e:
    print(f"New code caught expected error: {e}")

data_file.seek(0) # איפוס מצביע הקובץ
result_new = np.loadtxt(data_file, dtype=float).astype(int)
print(result_new)
""",
            "expected_old_error": "ValueError" # 1.23.0+ מעלה שגיאה.
        },
        {
            "name": "25. np.convolve with inexact mode string (Pre 1.21.0)",
            "old_code": """
import numpy as np
a = np.array([1, 2, 3])
v = np.array([0, 1, 0.5])
# בגרסאות ישנות (לפני 1.21.0), שימוש במחרוזת מצב לא מדויקת עשוי היה לעבוד (למשל, "Full" במקום "full").
result = np.convolve(a, v, mode='Full')
print(result)
""",
            "new_code": """
import numpy as np
a = np.array([1, 2, 3])
v = np.array([0, 1, 0.5])
# בגרסאות חדשות (1.21.0+), mode דורש התאמה מדויקת (למשל, "full").
try:
    result = np.convolve(a, v, mode='Full')
except ValueError as e:
    print(f"New code caught expected error: {e}")
result_new = np.convolve(a, v, mode='full') # שימוש במצב מדויק
print(result_new)
""",
            "expected_old_error": "ValueError" # 1.21.0+ מעלה שגיאה אם mode אינו מדויק.
        },
        {
            "name": "26. Removal of np.int alias (Pre 1.20.0)",
            "old_code": """
import numpy as np
# בגרסאות ישנות, ניתן היה להשתמש בכינוי זה.
dt = np.int
print(dt)
""",
            "new_code": """
import numpy as np
# בגרסאות חדשות (1.20.0+), יש להשתמש בסוג המובנה של Python.
dt = int
print(dt)
""",
            "expected_old_error": "AttributeError" # np.int הוסר וגרם לשגיאה.
        },
        {
            "name": "34. torch.Tensor.data Deprecation (Use .detach())",
            "old_code": """
import torch

x = torch.randn(2, 2, requires_grad=True)
# Accessing .data directly is deprecated as it bypasses autograd.
y_old = x.data * 2
print(y_old)
""",
            "new_code": """
import torch

x = torch.randn(2, 2, requires_grad=True)
# Use .detach() to get a view of the tensor that is detached from the computation graph.
y_new = x.detach() * 2
print(y_new)
""",
            "expected_old_error": "UserWarning" # Expected to raise a UserWarning about .data deprecation.
        },
        {
            "name": "35. Indexing with torch.uint8 tensors (Use torch.bool)",
            "old_code": """
import torch

x = torch.randn(3, 3)
mask_uint8 = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.uint8)
# Indexing with torch.uint8 is deprecated and will raise a TypeError.
result = x[mask_uint8]
print(result)
""",
            "new_code": """
import torch

x = torch.randn(3, 3)
mask_bool = torch.tensor([[False, True, False], [True, False, True], [False, True, False]], dtype=torch.bool)
# Use torch.bool tensors for boolean indexing.
result = x[mask_bool]
print(result)
""",
            "expected_old_error": "TypeError" # Indexing with uint8 raises TypeError.
        },
        {
            "name": "36. DataFrame.append() Deprecation (Use pd.concat())",
            "old_code": """
import pandas as pd

df1 = pd.DataFrame([['a', 1]], columns=['letter', 'number'])
df2 = pd.DataFrame([['b', 2]], columns=['letter', 'number'])
# DataFrame.append() is deprecated.
result = df1.append(df2, ignore_index=True)
print(result)
""",
            "new_code": """
import pandas as pd

df1 = pd.DataFrame([['a', 1]], columns=['letter', 'number'])
df2 = pd.DataFrame([['b', 2]], columns=['letter', 'number'])
# Use pd.concat() for combining DataFrames.
result = pd.concat([df1, df2], ignore_index=True)
print(result)
""",
            "expected_old_error": "FutureWarning" # Expected to raise FutureWarning.
        },
        {
            "name": "37. DataFrame.applymap() Deprecation (Use .map() or .apply())",
            "old_code": """
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# DataFrame.applymap() is deprecated.
result = df.applymap(lambda x: x * 2)
print(result)
""",
            "new_code": """
import pandas as pd

df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
# Use .map() for Series or .apply(..., axis=None) for element-wise operations on DataFrame.
result = df.apply(lambda x: x * 2, axis=None)
print(result)
""",
            "expected_old_error": "FutureWarning" # Expected to raise FutureWarning.
        },
        {
            "name": "38. DataFrame.backfill() Deprecation (Use .bfill())",
            "old_code": """
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, np.nan]})
# DataFrame.backfill() is deprecated and will raise an AttributeError.
result = df.backfill()
print(result)
""",
            "new_code": """
import pandas as pd
import numpy as np

df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [np.nan, 5, np.nan]})
# Use DataFrame.bfill() instead of backfill().
result = df.bfill()
print(result)
""",
            "expected_old_error": "AttributeError" # Expected to raise AttributeError.
        },
        {
            "name": "39. DataFrame.bool() Deprecation (Use .item())",
            "old_code": """
import pandas as pd

s = pd.Series([True])
# DataFrame.bool() (or Series.bool()) is deprecated.
# This will raise a TypeError in newer versions.
result = s.bool()
print(result)
""",
            "new_code": """
import pandas as pd

s = pd.Series([True])
# Use .item() to extract the Python scalar from a single-element Series/DataFrame.
result = s.item()
print(result)
""",
            "expected_old_error": "TypeError" # Expected to raise TypeError.
        },
        {
            "name": "40. Implicit Downcasting Deprecation (FutureWarning)",
            "old_code": """
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
# Operations like replace() with mixed types might implicitly downcast.
# This will raise a FutureWarning in modern pandas.
df['A'] = df['A'].replace(2, 'new')
print(df)
print(df.dtypes)
""",
            "new_code": """
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
# Explicitly handle dtypes after operations to avoid implicit downcasting.
df['A'] = df['A'].replace(2, 'new').infer_objects(copy=False)
print(df)
print(df.dtypes)
""",
            "expected_old_error": "FutureWarning" # Expected to raise FutureWarning.
        },
        {
            "name": "41. mode.copy_on_write Option Deprecation (FutureWarning)",
            "old_code": """
import pandas as pd

# Setting mode.copy_on_write is deprecated as CoW will be default.
pd.set_option("mode.copy_on_write", True)
print("mode.copy_on_write set (old method).")
""",
            "new_code": """
import pandas as pd

# Copy-on-Write will be the default in pandas 3.0. This option is no longer needed.
print("Copy-on-Write will be default in pandas 3.0.")
""",
            "expected_old_error": "FutureWarning" # Expected to raise FutureWarning.
        },
        {
            "name": "42. DataFrame.bool() Deprecation (Use DataFrame.item())",
            "old_code": """
import pandas as pd

df_single = pd.DataFrame([[True]])
# df.bool() is deprecated and will raise a TypeError.
try:
    result = df_single.bool()
    print(result)
except Exception as e:
    print(f"Old code failed as expected with error: {type(e).__name__}: {e}")
""",
            "new_code": """
import pandas as pd

df_single = pd.DataFrame([[True]])
# Use df.item() to extract the scalar value.
result = df_single.item()
print(result)
""",
            "expected_old_error": "TypeError" # Expected to raise TypeError.
        },
        {
            "name": "43. Implicit Downcasting Deprecation in Pandas (FutureWarning)",
            "old_code": """
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
# Operations like fillna with different dtypes can lead to implicit downcasting.
# This will raise a FutureWarning in modern pandas.
df_old = df.copy()
df_old['A'] = df_old['A'].fillna('x')
print(df_old)
print(df_old.dtypes)
""",
            "new_code": """
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
# Explicitly handle dtypes after operations to avoid implicit downcasting.
df_new = df.copy()
df_new['A'] = df_new['A'].fillna('x').infer_objects(copy=False)
print(df_new)
print(df_new.dtypes)
""",
            "expected_old_error": "FutureWarning" # Expected to raise FutureWarning.
        },
        {
            "name": "44. mode.copy_on_write Option Deprecation (FutureWarning)",
            "old_code": """
import pandas as pd

# Setting mode.copy_on_write is deprecated as CoW will be default in pandas 3.0.
# This will raise a FutureWarning.
pd.set_option("mode.copy_on_write", True)
print("mode.copy_on_write set (old method).")
""",
            "new_code": """
import pandas as pd

# Copy-on-Write (CoW) will be the default behavior in pandas 3.0.
# The option to toggle it is no longer needed.
print("Copy-on-Write will be default in pandas 3.0.")
""",
            "expected_old_error": "FutureWarning" # Expected to raise FutureWarning.
        },
    ]

    for example in all_code_examples:
        print(f"--- Testing: {example['name']} ---\n", flush=True)
        
        print("\n--- Old Code ---", flush=True)
        returncode, stdout, stderr = run_code_snippet(example['old_code'])
        if returncode != 0:
            print(f"Old code failed as expected with error:\n{stderr}\n", flush=True)
            if example["expected_old_error"] in stderr:
                print(f"  (Expected error or failure type '{example['expected_old_error']}' found in stderr.)\n", flush=True)
            # הערה: עבור כשלי לוגיקה, הפלט הרגיל של הקוד הישן יהיה שגוי.
        else:
            print(f"Old code unexpectedly succeeded:\n{stdout}\n", flush=True)
        
        print("\n--- New Code ---", flush=True)
        returncode, stdout, stderr = run_code_snippet(example['new_code'])
        if returncode == 0:
            print(f"New code succeeded as expected:\n{stdout}\n", flush=True)
        else:
            print(f"New code unexpectedly failed:\n{stderr}\n", flush=True)
        
        print("\n" + "="*50 + "\n", flush=True)